import torch.multiprocessing as multiprocessing
from tqdm import tqdm
from collections import Counter
import torch
from torch import nn
import numpy as np
from .clonaltrans import CloneTranModel
import time
import pandas as pd
import os
from torch.nn.parameter import Parameter
from natsort import natsorted
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from matplotlib.lines import Line2D
from .utils import set_seed

class Bootstrapping(nn.Module):
    def __init__(self, model, offset: int = 0) -> None:
        super(Bootstrapping, self).__init__()

        self.t_observed = model.t_observed.to('cpu')
        self.data_dir = model.data_dir
        self.config = model.config
        self.num_gpus = 30

        self.N = model.N.detach().clone().to('cpu')
        self.L = model.L.detach().clone().to('cpu')
        self.config = model.config
        self.num_gpus = 4
        self.offset = offset

    def bootstart(self, num_boots=100):
        print (time.asctime())
        print (f'# of bootstrapping trails: {num_boots}, # of pseudo GPUs used: {self.num_gpus}')
        
        assert num_boots % self.num_gpus == 0
        multiprocessing.set_start_method('spawn')

        pbar = tqdm(range(int(num_boots / self.num_gpus)))

        for epoch in pbar:
            with multiprocessing.Pool(self.num_gpus) as pool:
                res = pool.map_async(
                    self.process, 
                    self.sample_replace(self.N, epoch)
                )
                
                pool.close()
                pool.join()

    def sample_replace(self, N_ori, epoch):
        buffer, tps, pops = [], N_ori.shape[0], N_ori.shape[2]
        indices = np.arange(0, tps * pops)
        indices_view = indices.reshape((tps, pops))

        for gpu_id in range(self.num_gpus):
            sample_N = torch.zeros(N_ori.shape)

            samples = np.random.choice(indices, tps * pops, replace=True)
            counter = Counter(samples)

            for tp in range(tps):
                for pop in range(pops):
                    pos = indices_view[tp][pop]

                    if pos in counter.keys():
                        sample_N[tp, :, pop] = counter[pos]

            sample_N[0, :, :] = 1
            buffer.append([sample_N, gpu_id % 2, epoch * self.num_gpus + gpu_id + self.offset])
        
        return buffer

    def process(self, args):
        sample_N, gpu_id, model_id = args
        set_seed(42)

        self.config.gpu = gpu_id
        self.config.learning_rate = 0.002
        self.config.num_epochs = 3000
        self.config.lrs_ms = [500 * i for i in range(1, 6)]

        model = CloneTranModel(
            N=self.N.to(gpu_id), 
            L=self.L.to(gpu_id), 
            config=self.config, 
            writer=None, 
            sample_N=sample_N.to(gpu_id)
        ).to(gpu_id)

        model.trainable = True
        model.t_observed = self.t_observed.clone().to(gpu_id)
        model.data_dir = self.data_dir
        model.model_id = model_id

        try:
            model.train_model(model.t_observed)
        except:
            model.trainable = False

        #TODO save only trainable & reasonable reconstruction loss models
        if model.trainable:
            model.ode_func = model.ode_func.to('cpu')
            model.input_N = model.input_N.to('cpu')
            model.oppo_L_nondia = model.oppo_L_nondia.to('cpu')
            torch.save(model, f'./dyna_boots10tps/{model.model_id}.pt')
    
    def boot_validate(self, model):
        pena_K1_nonL, _, pena_pop_zero, _ = model.get_matrix_K(K_type=model.config.K_type)
        
        if torch.sum(pena_K1_nonL) < 0.5:
            return model
        else:
            return None

class ProfileLikelihood(nn.Module):
    def __init__(self, model, model_path) -> None:
        super(ProfileLikelihood, self).__init__()

        self.t_observed = model.t_observed.to('cpu')
        self.data_dir = model.data_dir
        self.config = model.config
        self.num_gpus = 30

        self.N = model.N.detach().clone().to('cpu')
        self.L = model.L.detach().clone().to('cpu')
        self.K1 = model.ode_func.K1.detach().clone().to('cpu')
        self.K2 = model.ode_func.K2.detach().clone().to('cpu')
        self.std = model.ode_func.std.detach().clone().to('cpu')

        self.paga = pd.read_csv(os.path.join(model.data_dir, 'graph_table.csv'), index_col=0).astype(np.int32).values
        np.fill_diagonal(self.paga, 1)

        self.anno = pd.read_csv(os.path.join(model.data_dir, 'annotations.csv')).values
        self.model_path = model_path

    def profilestart(self):
        print (time.asctime())
        print (f'# of pseudo GPUs used: {self.num_gpus}')
        multiprocessing.set_start_method('spawn')

        for clone in range(self.N.shape[1]):
            for pop1 in range(self.N.shape[2]):
                for pop2 in range(self.N.shape[2]):

                    if self.paga[pop1, pop2] == 1:
                        best_fit = self.K1[clone, pop1, pop2] if pop1 != pop2 else self.K2[clone, pop1]
                        print ('Profile parameter for clone {}, pop1 {}, pop2 {}, value of best fit {:.3f}'.format(clone, self.anno[pop1, 1], self.anno[pop2, 1], best_fit.item()))

                        with multiprocessing.Pool(self.num_gpus) as pool:
                            res = pool.map_async(
                                self.profile_process, 
                                self.fix_para_model(clone, pop1, pop2, best_fit)
                            )
                            
                            pool.close()
                            pool.join()
                        
                        dir_models = os.path.join(os.path.split(self.model_path)[0], '..', 'profilelikeli')
                        ref_model = torch.load(os.path.join(dir_models, f'C{clone}_P{pop1}_P{pop2}_T15.pt'))
                        cal_K, likeli = self.profile_validate(
                            ref_model, clone, pop1, pop2, 
                            dir_models=dir_models
                        )
                        self.plot_profile(cal_K, likeli, clone, self.anno[pop1, 1], self.anno[pop2, 1])

    def fix_para_model(self, clone, pop1, pop2, best_fit):
        buffer = []
        candidates = torch.linspace(best_fit.item() - 0.5, best_fit.item() + 0.5, self.num_gpus + 1)

        for trail_id, candi in enumerate(candidates):
            buffer.append([candi, trail_id % 3 + 1, clone, pop1, pop2, trail_id])

        return buffer

    def profile_process(self, args):
        candi, gpu_id, clone, pop1, pop2, trail_id = args
        set_seed(42)

        self.config.gpu = gpu_id
        self.config.learning_rate = 0.002
        self.config.num_epochs = 2000
        self.config.lrs_ms = [500 * i for i in range(1, 4)]

        supplement = [
            torch.ones((self.N.shape[1], self.N.shape[2], self.N.shape[2])), torch.zeros((self.N.shape[1], self.N.shape[2], self.N.shape[2])),
            torch.ones((self.N.shape[1], self.N.shape[2])), torch.zeros((self.N.shape[1], self.N.shape[2])),
        ]

        if pop1 != pop2:
            supplement[0][clone, pop1, pop2] = 0
            supplement[1][clone, pop1, pop2] = candi
        else:
            supplement[2][clone, pop1] = 0
            supplement[3][clone, pop1] = candi

        self.supplement = [Parameter(supplement[i], requires_grad=False).to(gpu_id) for i in range(4)]
        model = CloneTranModel(
            N=self.N.to(gpu_id), 
            L=self.L.to(gpu_id), 
            config=self.config, 
            writer=None, 
            sample_N=torch.ones(self.N.shape).to(gpu_id),
            extras=[self.std.clone().to(gpu_id), self.K1.clone().to(gpu_id), self.K2.clone().to(gpu_id), self.supplement]
        ).to(gpu_id)

        model.trainable = True
        model.t_observed = self.t_observed.clone().to(gpu_id)
        model.data_dir = self.data_dir
        model.model_id = f'C{clone}_P{pop1}_P{pop2}_T{trail_id}'

        try:
            model.train_model(model.t_observed)
        except:
            model.trainable = False

        if model.trainable:
             model.input_N = model.input_N.to('cpu')
             model.ode_func = model.ode_func.to('cpu')
             model.ode_func.supplement = [i.to('cpu') for i in model.ode_func.supplement]
             torch.save(model.to('cpu'), os.path.join(os.path.split(self.model_path)[0], '..', 'profilelikeli', f'{model.model_id}.pt'))

    def get_likelihood(self, model, eps: float = 1e-3):
        std = torch.sqrt(torch.square(model.ode_func.std))
        dist = torch.distributions.Normal(
            loc=model.eval_model(model.t_observed), 
            scale=torch.maximum(std, torch.tensor([eps]))
        )
        log_prob = dist.log_prob(model.input_N)
        return -torch.sum(log_prob).detach().cpu().numpy()

    def profile_validate(
        self, 
        ref_model, 
        clone: int = 0, 
        pop1: int = 0, 
        pop2: int = 0, 
        eps: float = 1e-3, 
        dir_models: str = './profilelikeli/'
    ):
        model_list, likeli, Ks = [], [], []

        for files in natsorted(os.listdir(dir_models)):
            if files.startswith(f'C{clone}_P{pop1}_P{pop2}') and files != f'C{clone}_P{pop1}_P{pop2}_T15.pt':
                model_list.append(torch.load(os.path.join(dir_models, files)))
        print (f'# of profiled attemps: {len(model_list)} for C{clone}_P{pop1}_P{pop2}')

        model_list.append(ref_model)
        model_list[-1].ode_func = model_list[-1].ode_func.to('cpu')
        model_list[-1].ode_func.supplement = [i.to('cpu') for i in model_list[-1].ode_func.supplement]
        
        for idx, model in enumerate(model_list):
            model.input_N = model.input_N.to('cpu')
            Ks.append(model.get_matrix_K(model.config.K_type, eval=True).detach().cpu().numpy())

        Ks = np.stack(Ks)
        cal_K = Ks[:, clone, pop1, pop2]

        for idx, model in tqdm(enumerate(model_list)):
            likeli.append(self.get_likelihood(model, eps=eps))
        return cal_K, likeli

    def plot_profile(self, cal_K, likeli, clone, pop1, pop2):
        res = dict(zip(cal_K[:-1], likeli[:-1]))
        res = dict(sorted(res.items()))

        t = list(res.keys())
        interp = interpolate.interp1d(t, list(res.values()), kind='quadratic')
        x = np.linspace(np.min(list(res.keys())), np.max(list(res.keys())), 500)

        plt.plot(t, interp(t), color='#2C6975', marker='*', linestyle='', markersize=7)
        plt.plot(x, interp(x), color='#2C6975')
        plt.plot(cal_K[-1], likeli[-1], marker='o', color='lightcoral', markersize=5)
        plt.axvline(cal_K[-1], linestyle='--', color='#929591')

        legend_elements = [
            Line2D([0], [0], color='#2C6975', lw=1.5), 
            Line2D([0], [0], marker='o', color='lightcoral', markersize=5, linestyle='')
        ]
        labels = ['Profiled Trails', 'Original Model']
        plt.legend(legend_elements, labels, loc='right', fontsize=10, bbox_to_anchor=(1.35, 0.5))
        plt.xlabel('Profiled transition rates of Clone {}'.format(clone))
        plt.ylabel('Gaussian NLL')
        plt.title(f'From {pop1} to {pop2}', fontsize=10)

        plt.savefig(os.path.join(os.path.split(self.model_path)[0], '../..', f'figures/profilefigs/C{clone}_{pop1}_{pop2}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def profile_all(self, ref_model, dir_models):
        for clone in range(self.N.shape[1]):
            for pop1 in range(self.N.shape[2]):
                for pop2 in range(self.N.shape[2]):

                    if self.paga[pop1, pop2] == 1:
                        best_fit = self.K1[clone, pop1, pop2] if pop1 != pop2 else self.K2[clone, pop1]
                        print ('Profile parameter for clone {}, pop1 {}, pop2 {}, value of best fit {:.3f}'.format(clone, self.anno[pop1, 1], self.anno[pop2, 1], best_fit.item()))

                        cal_K, likeli = self.profile_validate(ref_model, clone, pop1, pop2, dir_models=dir_models)
                        self.plot_profile(cal_K, likeli, clone, self.anno[pop1, 1], self.anno[pop2, 1])