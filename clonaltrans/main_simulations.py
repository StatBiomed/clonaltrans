import torch 
from utils import set_seed
from trainer import CloneTranModel
from model import ODESolver
import numpy as np
import pandas as pd
import os
import torch.multiprocessing as multiprocessing
from tqdm import tqdm

from data import cordblood_data, cordblood_time, cordblood_simulation_N
from data import weinreb_data, weinreb_time, weinreb_simulation_N

def run_model(
    config, 
    N, 
    L, 
    t_simu, 
    save_name='model_last.pt'
):
    set_seed(config['system']['seed'])
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
        writer=None,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        t_observed=t_simu.to(config['system']['gpu_id'], dtype=torch.float32),
        trainer_type='simulations',
        sample_N=None,
        gpu_id=config['system']['gpu_id']
    )
    trainer.train_model()

    torch.save(trainer, save_name)

def sample_replace(epoch, used_gpu_ids, concurrent):
    buffer = []

    for pseudo_gpu_id in range(concurrent):
        assigned_gpu = used_gpu_ids[int(pseudo_gpu_id % len(used_gpu_ids))]
        assigned_id = epoch * concurrent + pseudo_gpu_id

        # t_simu, scaling_factor = cordblood_time(pseudo_gpu_id + 2)
        # model_const, model_dyna, K = cordblood_data()
        # N_simu = cordblood_simulation_N(K, model_const, t_simu)

        # t_simu, scaling_factor = cordblood_time(4)
        # model_const, model_dyna, K = cordblood_data()
        # N_simu = cordblood_simulation_N(K, model_const, t_simu, noise_level=assigned_id)

        # t_simu, scaling_factor = weinreb_time(pseudo_gpu_id + 2)
        # model_const, model_dyna = weinreb_data()
        # N_simu = weinreb_simulation_N(model_dyna, t_simu)

        t_simu, scaling_factor = weinreb_time(4)
        model_const, model_dyna = weinreb_data()
        N_simu = weinreb_simulation_N(model_dyna, t_simu, noise_level=assigned_id)

        # model = model_const if epoch % 2 == 0 else model_dyna
        model = model_dyna
        # model = model_const
        
        model.config['user_trainer']['scaling_factor'] = scaling_factor
        model.config['system']['gpu_id'] = assigned_gpu

        buffer.append([
            model.config, 
            N_simu.detach().cpu(), 
            model.L.detach().cpu(), 
            t_simu.to(assigned_gpu), 
            assigned_id
        ])
    
    return buffer

def process(bootstrap_args):
    config, N_simu, L, t_simu, assigned_id = bootstrap_args

    # config._save_dir = f"/ssd/users/mingzegao/clonaltrans/trials/revision_1/checkpoints/SimulationsConst/{config['arch']['args']['K_type']}_{len(config['user_trainer']['scaling_factor'])}"
    # config._save_dir = f"/ssd/users/mingzegao/clonaltrans/trials/revision_1/checkpoints/SimulationsDynamic/{config['arch']['args']['K_type']}_{len(config['user_trainer']['scaling_factor'])}"
    # config._save_dir = f"/ssd/users/mingzegao/clonaltrans/trials/revision_1/checkpoints/SimulationsConstNoise/{assigned_id}"
    config._save_dir = f"/ssd/users/mingzegao/clonaltrans/trials/revision_1/checkpoints/SimulationsDynamicNoise/{assigned_id}"
    os.makedirs(config._save_dir, exist_ok=True)

    print (os.path.join(config._save_dir, f"model_last.pt"), torch.mean(N_simu))

    run_model(
        config, 
        N_simu, 
        L, 
        t_simu, 
        save_name=os.path.join(config._save_dir, f"model_last.pt")
    )

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    pbar = tqdm(range(2))
    concurrent = 24

    with multiprocessing.Pool(concurrent) as pool:
        for epoch in pbar:
            for res in pool.imap_unordered(
                process, 
                sample_replace(epoch, [4, 5, 6, 7], concurrent)
            ):
                pass