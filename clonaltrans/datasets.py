import torch
import pandas as pd
import os
import numpy as np
from torchdiffeq import odeint
from torch.nn.parameter import Parameter

def get_topo_obs(
    data_dir, 
    fill_diagonal: bool = False, 
    init_day_zero: bool = True,
    device: int = 0
):
    # PAGA Topology (Population Adjacency Matrix), same for each clones?
    paga = pd.read_csv(os.path.join(data_dir, 'graph_table.csv'), index_col=0).astype(np.int32)
    print (f'Topology graph loaded {paga.shape}.')

    # Will be filled when initializing model, ignore for now
    if fill_diagonal:
        np.fill_diagonal(paga.values, 1)

    # number of cells (per timepoints, per meta-clone, per population)
    array = np.loadtxt(os.path.join(data_dir, 'kinetics_array_correction_factor.txt'))
    array_ori = array.reshape(array.shape[0], array.shape[1] // 11, 11)
    array_ori = torch.swapaxes(torch.tensor(array_ori, dtype=torch.float32), 0, 1)
    print (f'Input cell data (num_timepoints {array_ori.shape[0]}, num_clones {array_ori.shape[1]}, num_populations {array_ori.shape[2]}) loaded.')

    if init_day_zero:
        init_con = pd.read_csv(os.path.join(data_dir, 'initial_condition.csv'), index_col=0).astype(np.float32)
        day_zero = np.zeros((array_ori.shape[1], array_ori.shape[2]))
        day_zero[:, 0] = init_con['leiden'].values
        array_ori = torch.concatenate((torch.tensor(day_zero, dtype=torch.float32).unsqueeze(0), array_ori), axis=0)
        print (f'Day 0 has been added. Input data shape: {array_ori.shape}')

    # generate background cells
    background = torch.mean(array_ori, axis=1).unsqueeze(1)
    array_total = torch.concatenate((array_ori, background), axis=1)
    print (f'Background reference cells generated (mean of all other clones). Input data shape: {array_total.shape}')

    return torch.tensor(paga.values, dtype=torch.float32, device=device), array_total.to(device)

def simulation(
    K, # (num_clones, num_populations, num_populations)
    L, # (num_populations, num_populations)
    config,
    t_simu=torch.tensor([0.0, 0.5, 1.0]), 
    y0=None,
    noise_level=1e-2
):
    '''
    Given rates, generate data at raw scale by adding noise. 
    Potential variables to vary: 
        1. Noise level
        2. Number of time points 
        3. Dynamic shapes (up & down) 
        4. (e.g., estimated from seed data)
    '''

    t_simu = (t_simu - t_simu[0]) / (t_simu[-1] - t_simu[0])
    L = torch.broadcast_to(L.unsqueeze(0), (K.shape[0], K.shape[1], K.shape[1]))

    from .ode_block import ODEBlock
    ode_func = ODEBlock(
        num_tpoints=4,
        num_clones=K.shape[0],
        num_pops=K.shape[1],
        hidden_dim=config.hidden_dim, 
        activation=config.activation, 
        num_layers=1
    )

    ode_func.K1 = Parameter(torch.sqrt(K * L), requires_grad=True)
    ode_func.K2 = Parameter(torch.diagonal(K, dim1=-2, dim2=-1), requires_grad=True)

    array_total = odeint(ode_func, y0, t_simu, rtol=1e-5, method='dopri5', options=dict(dtype=torch.float32))

    scale_factor = array_total.abs()
    scale_factor[torch.rand_like(scale_factor) < 0.5] *= -1
    array_total = array_total + scale_factor * torch.rand_like(array_total) * noise_level

    return array_total.to(config.gpu)