import argparse
import torch 
from utils import ConfigParser
from typing import List
import numpy as np
import os

import data.datasets as module_data
from utils import set_seed
from torch.utils.tensorboard import SummaryWriter
from trainer import CloneTranModel
from model import ODESolver

def run_model(config):
    set_seed(config['system']['seed'])
    writer = SummaryWriter(log_dir=config._log_dir)
    
    logger = config.get_logger('train')
    logger.info('Estimating clonal specific transition rates.\n')
    config['data_loader']['args']['logger'] = logger

    data_loader = config.init_obj('data_loader', module_data)
    N, L = data_loader.get_datasets()
    N, L = N.to(config['system']['gpu_id']), L.to(config['system']['gpu_id'])

    model = ODESolver(
        L=L,
        num_clones=N.shape[1],
        num_pops=N.shape[2],
        hidden_dim=config['arch']['args']['hidden_dim'], 
        activation=config['arch']['args']['activation'], 
        K_type=config['arch']['args']['K_type'],
        adjoint=config['user_trainer']['adjoint']
    ).to(config['system']['gpu_id'])
    
    logger.info(model)
    logger.info('Integration time of ODE solver is {}'.format(config['user_trainer']['t_observed']))

    params = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    logger.info('Trainable parameters: {}\n'.format(params))

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['optimizer']['learning_rate'], 
        weight_decay=config['optimizer']['weight_decay'],
        amsgrad=True
    )
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['learning_rate'], amsgrad=True)

    trainer = CloneTranModel(
        N=N, 
        L=L, 
        config=config,
        model=model,
        optimizer=optimizer,
        t_observed=torch.tensor(config['user_trainer']['t_observed']).to(config['system']['gpu_id'], dtype=torch.float32),
        trainer_type='training',
        writer=writer,
        sample_N=None,
        gpu_id=config['system']['gpu_id']
    )
    trainer.train_model()

    trainer.writer.flush()
    trainer.writer.close()
    torch.save(trainer, os.path.join(config.save_dir, 'model_last.pt'))

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Estimation of Clonal Transition Rates')
    args.add_argument('-c', '--config', default='config.json', type=str, help='config file path (default: None)')
    args.add_argument('-id', '--run_id', default=None, type=str, help='id of experiment (default: current time)')

    config = ConfigParser.from_args(args)
    run_model(config)