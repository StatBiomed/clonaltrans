import torch.multiprocessing as multiprocessing
from tqdm import tqdm
from collections import Counter
import torch
from torch import nn
import numpy as np
from ..trainer.clonaltrans import CloneTranModel
import time
import pandas as pd
import os
from torch.nn.parameter import Parameter
from natsort import natsorted
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from matplotlib.lines import Line2D
from ..utils import set_seed

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

        from itertools import product
        for clone, pop1, pop2 in product(range(self.N.shape[1]), range(self.N.shape[2]), range(self.N.shape[2])):
        # for clone in range(self.N.shape[1]):
        #     for pop1 in range(self.N.shape[2]):
        #         for pop2 in range(self.N.shape[2]):

            if self.paga[pop1, pop2] == 1:
                best_fit = self.K1[clone, pop1, pop2] if pop1 != pop2 else self.K2[clone, pop1]
                print ('Profile parameter for clone {}, pop1 {}, pop2 {}, value of best fit {:.3f}'.format(clone, self.anno[pop1, 1], self.anno[pop2, 1], best_fit.item()))

                with multiprocessing.Pool(self.num_gpus + 1) as pool:
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
        if pop1 == pop2:
            candidates = torch.linspace(best_fit.item() - 0.3, best_fit.item() + 0.3, self.num_gpus + 1)
        if pop1 != pop2:
            best_fit = torch.abs(best_fit)
            candidates = torch.linspace(best_fit.item() - 0.3, best_fit.item() + 0.3, self.num_gpus + 1)
            candidates = torch.clamp(candidates, 0.0, candidates[-1])

        for trail_id, candi in enumerate(candidates):
            buffer.append([candi, trail_id % 3 + 1, clone, pop1, pop2, trail_id])

        return buffer

    def profile_process(self, args):
        candi, gpu_id, clone, pop1, pop2, trail_id = args
        set_seed(42)

        self.config.gpu = gpu_id
        self.config.learning_rate = 0.001
        self.config.num_epochs = 1000
        self.config.lrs_ms = [300 * i for i in range(1, 4)]

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
                model_list.append(torch.load(os.path.join(dir_models, files), map_location='cpu'))
        print (f'# of profiled attemps: {len(model_list)} for C{clone}_P{pop1}_P{pop2}')

        model_list.append(torch.load(os.path.join(dir_models, f'C{clone}_P{pop1}_P{pop2}_T15.pt'), map_location='cpu'))
        model_list[-1].ode_func = model_list[-1].ode_func.to('cpu')
        model_list[-1].ode_func.supplement = [i.to('cpu') for i in model_list[-1].ode_func.supplement]
        
        for idx, model in enumerate(model_list):
            model.input_N = model.input_N.to('cpu')
            Ks.append(model.get_matrix_K(model.config.K_type, eval=True).detach().cpu().numpy())

        Ks = np.stack(Ks)
        cal_K = Ks[:, clone, pop1, pop2]

        for idx, model in enumerate(model_list):
            likeli.append(self.get_likelihood(model, eps=eps))
        return cal_K, likeli

    def plot_profile(self, cal_K, likeli, clone, pop1, pop2):
        res = dict(zip(cal_K[:-1], likeli[:-1]))
        res = dict(sorted(res.items()))

        t = list(res.keys())
        interp = interpolate.interp1d(t, list(res.values()), kind='linear')
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
        # plt.legend(legend_elements, labels, loc='right', fontsize=10, bbox_to_anchor=(1.35, 0.5))
        plt.xlabel('Profiled transition rates of Clone {}'.format(clone), fontsize=14)
        plt.ylabel('Gaussian NLL', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.title(f'From {pop1} to {pop2}', fontsize=10)

        # if np.max(likeli) - np.min(likeli) < 1:
        #     plt.ylim([194, 198])
        # plt.ylim([likeli[15] - 1, likeli[15] + 3.8])

        # plt.savefig(os.path.join(os.path.split(self.model_path)[0], '../..', f'figures/profilefigs/C{clone}_{pop1}_{pop2}.svg'), dpi=600, bbox_inches='tight', transparent=False, facecolor='white')
        plt.savefig(f'./C{clone}_{pop1}_{pop2}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')
        plt.close()

import argparse
import torch 
from utils import ConfigParser
from typing import List
import numpy as np

import data.datasets as module_data
from utils import set_seed
from torch.utils.tensorboard import SummaryWriter
from trainer import CloneTranModel
from model import ODEBlock

def run_model(
    config, 
    N: any = None,
    L: any = None
):
    set_seed(config['seed'])
    writer = SummaryWriter(log_dir=config._log_dir)
    
    logger = config.get_logger('train')
    logger.info('Estimating clonal specific transition rates.\n')
    config['data_loader']['args']['logger'] = logger

    if N == None:
        data_loader = config.init_obj('data_loader', module_data)
        N, L = data_loader.get_datasets()
        N, L = N.to(config['gpu_id']), L.to(config['gpu_id'])

    model = ODEBlock(
        L=L,
        num_clones=N.shape[1],
        num_pops=N.shape[2],
        hidden_dim=config['arch']['args']['hidden_dim'], 
        activation=config['arch']['args']['activation'], 
        K_type=config['arch']['args']['K_type']
    ).to(config['gpu_id'])
    model.supplement = [item.to(config['gpu_id']) for item in model.supplement]
    
    logger.info(model)
    logger.info('Integration time of ODE solver is {}'.format(config['user_trainer']['t_observed']))

    params = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    logger.info('Trainable parameters: {}\n'.format(params))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['learning_rate'], amsgrad=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['optimizer']['lrs_ms'], gamma=0.5)

    trainer = CloneTranModel(
        N=N, 
        L=L, 
        config=config,
        writer=writer,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        t_observed=torch.tensor(config['user_trainer']['t_observed']).to(config['gpu_id'], dtype=torch.float32)
    )
    trainer.train_model()

    trainer.writer.flush()
    trainer.writer.close()
    torch.save(trainer, config.save_dir)
    return trainer

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Estimation of Clonal Transition Rates')
    args.add_argument('-c', '--config', default='config.json', type=str, help='config file path (default: None)')
    args.add_argument('-id', '--run_id', default=None, type=str, help='id of experiment (default: current time)')

    config = ConfigParser.from_args(args)
    run_model(config)