import multiprocessing
from tqdm import tqdm
from collections import Counter
import torch
from torch import nn
import numpy as np
from .clonaltrans import CloneTranModel
import time

class Bootstrapping(nn.Module):
    def __init__(self, model) -> None:
        super(Bootstrapping).__init__()

        self.N = model.N
        self.L = model.L
        self.config = model.config
        self.t_observed = model.t_observed
        self.data_dir = model.data_dir
        self.num_gpus = 4

    def bootstart(self, num_boots=100):
        print (time.asctime())
        print (f'# of bootstrapping trails: {num_boots}, # of pseudo GPUs used: {self.num_gpus}')
        
        assert num_boots % self.num_gpus == 0
        multiprocessing.set_start_method('spawn')

        pbar = tqdm(range(int(num_boots / self.num_gpus)))

        for epoch in pbar:
            self.epoch = epoch
            with multiprocessing.Pool(self.num_gpus) as pool:
                res = pool.map_async(
                    self.process, 
                    self.sample_replace(self.N.clone())
                ).get()
        
        print (time.asctime())

    def sample_replace(self, N_ori):
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

            sample_N[0, :, 0] = 1
            buffer.append([sample_N, gpu_id % 4, self.epoch * self.num_gpus + gpu_id])
        
        return buffer

    def process(self, args):
        sample_N, gpu_id, model_id = args

        self.config.gpu = gpu_id
        self.config.num_epochs = 2000
        self.config.lrs_ms = [500 * i for i in range(1, 4)]

        model = CloneTranModel(
            N=self.N.clone().to(gpu_id), 
            L=self.L.clone().to(gpu_id), 
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
            # model.input_N = model.input_N.to('cpu')
            torch.save(model.to('cpu'), f'./tempnormday0/{model.model_id}.pt')

        return None
