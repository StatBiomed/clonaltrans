import torch
import os 
from .datasets_simulations import simulation_const, simulation_dyna

def cordblood_data():
    os.chdir('/ssd/users/mingzegao/clonaltrans/clonaltrans/')

    path = '../trials/revision_1/checkpoints/CordBloodConstRates/0929_115228/model_last.pt'
    model_const = torch.load(path, map_location='cpu')

    path = '../trials/revision_1/checkpoints/CordBloodDynamicRates/0929_115437/model_last.pt'
    model_dyna = torch.load(path, map_location='cpu')

    device = torch.device('cpu')
    os.chdir('/ssd/users/mingzegao/clonaltrans/')

    K = model_const.get_matrix_K(eval=True)

    return model_const, model_dyna, K

def weinreb_data():
    os.chdir('/ssd/users/mingzegao/clonaltrans/clonaltrans/')

    path = '../trials/revision_1/checkpoints/WeinrebConstRates/0929_115327/model_last.pt'
    model_const = torch.load(path, map_location='cpu')

    path = '../trials/revision_1/checkpoints/WeinrebDynamicRates/0929_115710/model_last.pt'
    model_dyna = torch.load(path, map_location='cpu')

    device = torch.device('cpu')
    os.chdir('/ssd/users/mingzegao/clonaltrans/')

    return model_const, model_dyna

def cordblood_simulation_N(K, model_const, t_simu, noise_level=None):
    N_simu = simulation_const(
        K,
        model_const.L.detach().cpu(),
        model_const.config,
        t_simu,
        model_const.N[0].detach().cpu(),
        noise_level=noise_level
    )
    return N_simu

def weinreb_simulation_N(model_dyna, t_simu, noise_level=None):
    K1_encode = model_dyna.model.block.K1_encode.detach()
    K1_decode = model_dyna.model.block.K1_decode.detach()
    K2_encode = model_dyna.model.block.K2_encode.detach()
    K2_decode = model_dyna.model.block.K2_decode.detach()

    N_simu = simulation_dyna(
        K1_encode, K1_decode, K2_encode, K2_decode, 
        model_dyna.L.detach().cpu(), 
        model_dyna.config,
        t_simu,
        model_dyna.N[0].detach().cpu(),
        noise_level=noise_level
    )
    return N_simu

def cordblood_time(tpoints=2):
    if tpoints == 2:
        t_simu = torch.tensor([0.0, 17.0])
        scaling_factor = [1, 508.6567]

    if tpoints == 3:
        t_simu = torch.tensor([0.0, 3.0, 17.0])
        scaling_factor = [1, 4.9054, 508.6567]

    if tpoints == 4:
        t_simu = torch.tensor([0.0, 3.0, 10.0, 17.0])
        scaling_factor = [1, 4.9054, 530.5978, 508.6567]

    if tpoints == 5:
        t_simu = torch.tensor([0.0, 3.0, 7.0, 10.0, 17.0])
        scaling_factor = [1, 4.9054, 260, 530.5978, 508.6567]

    if tpoints == 6:
        t_simu = torch.tensor([0.0, 3.0, 7.0, 10.0, 14.0, 17.0])
        scaling_factor = [1, 4.9054, 260, 530.5978, 515, 508.6567]
    
    return t_simu, scaling_factor

def weinreb_time(tpoints=2):
    if tpoints == 2:
        t_simu = torch.tensor([0.0, 8.0])
        scaling_factor = [1, 12]

    if tpoints == 3:
        t_simu = torch.tensor([0.0, 4.0, 8.0])
        scaling_factor = [1, 3.1616, 12]

    if tpoints == 4:
        t_simu = torch.tensor([0.0, 2.0, 6.0, 8.0])
        scaling_factor = [1, 1.0808, 8.7108, 12]

    if tpoints == 5:
        t_simu = torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0])
        scaling_factor = [1, 1.0808, 3.1616, 8.7108, 12]
    
    return t_simu, scaling_factor