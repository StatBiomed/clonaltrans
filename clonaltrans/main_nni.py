import argparse
import torch 
from utils import ConfigParser
import numpy as np
import os
import logging
import torch.multiprocessing as multiprocessing
from tqdm import tqdm
from pathlib import Path
import json

from utils import set_seed
from torch.utils.tensorboard import SummaryWriter
from trainer import CloneTranModel
from model import ODESolver
from logger import setup_logging

import data.datasets as module_data
from data.nni import convert_to_serializable, prepare_nni, get_nni_combinations
from data.datasets_manuscript import prepare_manuscript_data

def run_model(config, logger):
    ''' <------- Prepare config for NNI -------> '''
    set_seed(config['system']['seed'])
    config = prepare_manuscript_data(config, logger, dataset=config['data_loader']['abbreviation'])
    ''' <------- Prepare config for NNI -------> '''

    writer = SummaryWriter(log_dir=config._log_dir)

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
        adjoint=config['user_trainer']['adjoint'],
        clipping=config['arch']['args']['clipping']
    ).to(config['system']['gpu_id'])
    
    logger.info(model)
    logger.info('Integration time of ODE solver is {}'.format(config['user_trainer']['t_observed']))

    params = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    logger.info('Trainable parameters: {}\n'.format(params))

    if config['optimizer']['scheduler_type'] == 'MultiStepLR':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['learning_rate'], amsgrad=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['optimizer']['multistep_milestone'], gamma=0.5)

    elif config['optimizer']['scheduler_type'] == 'AutoAdaptive':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['learning_rate'], weight_decay=0.0, amsgrad=True)
        scheduler = None
    
    else:
        raise ValueError('Invalid scheduler_type, please choose from AutoAdaptive or MultiStepLR')

    trainer = CloneTranModel(
        N=N, 
        L=L, 
        config=config,
        writer=writer,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        t_observed=torch.tensor(config['user_trainer']['t_observed']).to(config['system']['gpu_id'], dtype=torch.float32),
        trainer_type='training',
        sample_N=None,
        gpu_id=config['system']['gpu_id']
    )
    trainer.train_model()

    trainer.writer.flush()
    trainer.writer.close()
    torch.save(trainer, os.path.join(config.save_dir, 'model_last.pt'))

def sample_replace(epoch, used_gpu_ids, config, concurrent):
    buffer = []

    for pseudo_gpu_id in range(concurrent):
        assigned_gpu = used_gpu_ids[int(pseudo_gpu_id % len(used_gpu_ids))]
        assigned_id = epoch * concurrent + pseudo_gpu_id

        # tunable_params = get_nni_combinations(np.arange(4, 129, 1)[assigned_id])
        # tunable_params = get_nni_combinations(np.arange(0.1, 5.1, 0.1)[assigned_id])
        tunable_params = get_nni_combinations()[assigned_id]

        buffer.append([
            tunable_params, 
            assigned_gpu, 
            config,
            assigned_id
        ])
    
    return buffer

def process(bootstrap_args):
    tunable_params, gpu_id, config, assigned_id = bootstrap_args

    config._save_dir = str(config._save_dir) + '_' + str(assigned_id)
    config._log_dir = str(config._log_dir) + '_' + str(assigned_id)

    os.makedirs(config._save_dir, exist_ok=False)
    os.makedirs(config._log_dir, exist_ok=False)
    config['system']['gpu_id'] = gpu_id

    setup_logging(Path(config._log_dir), config.config['logger_config_path'])   

    logger = logging.getLogger(str(assigned_id))
    logger.setLevel(logging.DEBUG)

    with open(os.path.join(config._save_dir, 'parameters.json'), 'w') as json_file:
        json.dump({k: convert_to_serializable(v) for k, v in tunable_params.items()}, json_file, indent=4)
    
    config = prepare_nni(config, tunable_params)

    try: run_model(config, logger)
    except Exception as e: logger.info(str(e))

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Estimation of Clonal Transition Rates')
    args.add_argument('-c', '--config', default='config.json', type=str, help='config file path (default: None)')
    args.add_argument('-id', '--run_id', default=None, type=str, help='id of experiment (default: current time)')

    config = ConfigParser.from_args(args)

    multiprocessing.set_start_method('spawn')
    pbar = tqdm(range(1))
    concurrent = 8

    with multiprocessing.Pool(concurrent) as pool:
        for epoch in pbar:
            for res in pool.imap_unordered(
                process, 
                sample_replace(epoch, [1, 2, 3, 5], config, concurrent)
            ):
                pass