import argparse
import torch 
from utils import ConfigParser
import numpy as np
import os
from typing import List

from utils import set_seed
from trainer import CloneTranModel
from model import ODESolver

import torch.multiprocessing as multiprocessing
from tqdm import tqdm
from collections import Counter
import torch
from torch import nn
import time
from itertools import product

class Bootstrapping(nn.Module):
    def __init__(
        self, 
        model, 
        config,
        logger
    ) -> None:
        super(Bootstrapping, self).__init__()

        self.model = model
        self.config = config
        self.logger = logger
        self.save_dir = config.save_dir

        os.mkdir(os.path.join(self.save_dir, 'models'))

        self.concurrent = config['concurrent']
        self.epoch = config['epoch']
        self.used_gpu_ids = config['system']['gpu_id']
        self.calibrate = config['calibrate']

        self.N = model.N.detach().clone().to('cpu')
        self.L = model.L.detach().clone().to('cpu')
        self.config = model.config

        if self.calibrate:
            self.error_queue = multiprocessing.Queue()

    def bootstart(self):
        self.logger.info(time.asctime())
        self.logger.info(f'# of epochs: {self.epoch}, # of pseudo GPUs used: {self.concurrent}')
        
        multiprocessing.set_start_method('spawn')
        pbar = tqdm(range(self.epoch))

        with multiprocessing.Pool(self.concurrent) as pool:
            for epoch in pbar:
                for res in pool.imap_unordered(
                    self.process, 
                    self.sample_replace(self.N, epoch)
                ):
                    pass

    def sample_replace(self, N_ori, epoch):
        buffer, tps, pops = [], N_ori.shape[0], N_ori.shape[2]
        indices = np.arange(0, tps * pops)
        indices_view = indices.reshape((tps, pops))

        for pseudo_gpu_id in range(self.concurrent):
            sample_N = torch.zeros(N_ori.shape)

            samples = np.random.choice(indices, tps * pops, replace=True)
            counter = Counter(samples)

            for tp, pop in product(range(tps), range(pops)):
                pos = indices_view[tp][pop]

                if pos in counter.keys():
                    sample_N[tp, :, pop] = counter[pos]

            sample_N[0, :, :] = 1

            assigned_gpu = self.used_gpu_ids[0] if len(self.used_gpu_ids) == 1 else self.used_gpu_ids[int(pseudo_gpu_id % len(self.used_gpu_ids))]
            buffer.append([sample_N, assigned_gpu, epoch * self.concurrent + pseudo_gpu_id])
        
        return buffer

    def process(self, args):
        sample_N, gpu_id, model_id = args

        model = ODESolver(
            L=self.L.to(gpu_id),
            num_clones=self.N.shape[1],
            num_pops=self.N.shape[2],
            hidden_dim=self.config['arch']['args']['hidden_dim'], 
            activation=self.config['arch']['args']['activation'], 
            K_type=self.config['arch']['args']['K_type'],
            adjoint=self.config['user_trainer']['adjoint']
        ).to(gpu_id)

        self.logger.info(f'Running model ID: {model_id}')
        self.logger.info('Integration time of ODE solver is {}'.format(self.config['user_trainer']['t_observed']))

        params = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
        self.logger.info('Trainable parameters: {}\n'.format(params))

        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.config['optimizer']['learning_rate'], 
            weight_decay=self.config['optimizer']['weight_decay'],
            amsgrad=True
        )

        trainer = CloneTranModel(
            N=self.N.to(gpu_id), 
            L=self.L.to(gpu_id), 
            config=self.config,
            model=model,
            optimizer=optimizer,
            t_observed=torch.tensor(self.config['user_trainer']['t_observed']).to(gpu_id, dtype=torch.float32),
            trainer_type='bootstrapping',
            writer=None,
            sample_N=sample_N.to(gpu_id),
            gpu_id=gpu_id
        )
        trainer.trainable = True
        trainer.model_id = model_id

        if self.calibrate:
            try: trainer.train_model()
            except Exception as e: self.error_queue.put(str(e))
    
            if not self.error_queue.empty():
                self.logger.info(self.error_queue.get())

        else:
            try: trainer.train_model()
            except: trainer.trainable = False

            if trainer.trainable:
                torch.save(trainer, os.path.join(self.save_dir, 'models', f'{trainer.model_id}.pt'))
        
        torch.cuda.empty_cache()

def run_model(config):
    set_seed(config['system']['seed'])
    
    logger = config.get_logger('bootstrap')
    logger.info('Estimating confidence intervals of parameters using bootstrapping method.\n')

    model_ori = torch.load(config['model_path'], map_location='cpu')
    boots = Bootstrapping(model_ori, config, logger)

    if config['calibrate']: boots.process([torch.ones(boots.N.shape), 0, 0])
    else: boots.bootstart()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Bootstrapping for confidence intervals')
    args.add_argument('-c', '--config', default='config.json', type=str, help='config file path (default: None)')
    args.add_argument('-id', '--run_id', default=None, type=str, help='id of experiment (default: current time)')

    config = ConfigParser.from_args(args)
    run_model(config)