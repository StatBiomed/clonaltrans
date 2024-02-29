import numpy as np
import os
import pandas as pd
import multiprocessing
from tqdm import tqdm
import torch
import argparse
from utils import ConfigParser, get_K_total
from model import gillespie_main
import time

class GillespieDivision():
    def __init__(
        self, 
        model, 
        config, 
        logger, 
        K_total=None, 
        time_all=None
    ) -> None:
        super(GillespieDivision, self).__init__()

        self.num_clones = K_total.shape[1]
        self.epoch = config['epoch']
        self.concurrent = config['concurrent']
        self.save_dir = config.save_dir
        self.config = config

        self.K_total = K_total # (num_time_points, num_clones, num_pops, num_pops)
        self.time_all = time_all
        self.logger = logger
        self.cluster_names = pd.read_csv(os.path.join(
            model.config['data_loader']['args']['data_dir'], 
            model.config['data_loader']['args']['annots']
        ))['populations'][:K_total.shape[2]]

        self.L = model.used_L.squeeze()
        self.L.fill_diagonal_(0)
        self.L = self.L.detach().cpu().numpy()

    def bootstart(self):
        multiprocessing.set_start_method('spawn')
        pbar = tqdm(range(self.num_clones))

        for clone in pbar:
            self.logger.info(f'Start multiprocessing for meta-clone {clone} at {time.asctime()}.')
            gillespie_dir = os.path.join(self.save_dir, f'clone_{clone}')
            if not os.path.exists(gillespie_dir):
                os.mkdir(gillespie_dir)

            with multiprocessing.Pool(self.concurrent) as pool:

                for epoch in range(self.epoch):
                    for res in pool.imap_unordered(
                        self.process, 
                        self.get_buffer(epoch, self.K_total[:, clone, :, :], gillespie_dir)
                    ):
                        pass
            
            self.logger.info(f'End multiprocessing for each meta-clone {clone} at {time.asctime()}.')

    def get_buffer(self, epoch, K_total, gillespie_dir):
        buffer = []
        for idx in range(self.concurrent):
            buffer.append([epoch * self.concurrent + idx, K_total, self.time_all, gillespie_dir])
        return buffer

    def process(self, args):
        seed, K_total, time_all, gillespie_dir = args
        gillespie_main(seed, K_total, time_all, self.cluster_names, gillespie_dir, self.config, self.L)

def run_model(config):
    logger = config.get_logger('gillespie')
    logger.info('Running Gillespie simulation algorithms.')
    logger.info('Preparing candidate transition rates for meta-clones.\n')

    model_ori = torch.load(config['model_path'], map_location='cpu')
    time_all = np.arange(model_ori.t_observed[0].cpu(), model_ori.t_observed[-1].cpu() + config['time_interval'], config['time_interval'])
    time_all = np.round(time_all, 3)
    
    K_total = get_K_total(model_ori, tpoints=time_all) 
    logger.info(f'Reference time points are {time_all[:5]}')
    logger.info(f'Dimension of transition rates for this Gillespie trail: {K_total.shape}')

    gilles = GillespieDivision(model_ori, config, logger, K_total, time_all)
    gilles.bootstart()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Bootstrapping for confidence intervals')
    args.add_argument('-c', '--config', default='config.json', type=str, help='config file path (default: None)')
    args.add_argument('-id', '--run_id', default=None, type=str, help='id of experiment (default: current time)')

    config = ConfigParser.from_args(args)
    run_model(config)