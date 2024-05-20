import numpy as np
import pandas as pd

import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.set_printoptions(suppress=True)

import os 
os.chdir('/ssd/users/mingzegao/clonaltrans/clonaltrans/')

path = '/ssd/users/mingzegao/clonaltrans/trails/checkpoints/CordBloodConstRates/0322_185005/model_last.pt'
model_const = torch.load(path, map_location='cpu')

path = '/ssd/users/mingzegao/clonaltrans/trails/checkpoints/CordBloodDynamicRates/0322_185055/model_last.pt'
model_dyna = torch.load(path, map_location='cpu')

device = torch.device('cpu')
os.chdir('/ssd/users/mingzegao/clonaltrans/')

from data import simulation_const
from main_simulations import run_model

K = model_const.get_matrix_K(eval=True)

model_const.config['user_trainer']['scaling_factor'] = [1, 5.3521, 300, 581.8464, 610, 634.0689]

N6 = simulation_const(
    K,
    model_const.L.detach().cpu(),
    model_const.config,
    model_const.config['user_trainer']['scaling_factor'],
    torch.tensor([0.0, 3.0, 7.0, 10.0, 14.0, 17.0]).to(device),
    model_const.N[0].detach().cpu()
)

model_dyna.config['system']['gpu_id'] = 4

model_dyna.config['user_trainer']['scaling_factor'] = [1, 5.3521, 300, 581.8464, 610, 634.0689]

run_model(
    model_dyna.config,
    N6.detach().cpu(),
    model_dyna.L.detach().cpu(),
    torch.tensor([0.0, 3.0, 7.0, 10.0, 14.0, 17.0]).to(device),
    save_name='./datasets/Simulations/const_to_dynamic_noise/time_6.pt'
)