import torch
from torchdiffeq import odeint
from torch.nn.parameter import Parameter

def get_array_total(ode_func, y0, t_simu, noise_level, scale=True, raw=False):
    array_total = odeint(ode_func, y0, t_simu, rtol=1e-4, atol=1e-4, method='dopri5', options=dict(dtype=torch.float32))

    if raw:
        return torch.round(array_total)

    if scale:
        scale_factor = array_total.abs()
        scale_factor[torch.rand_like(scale_factor) < 0.5] *= -1
        array_total = array_total + scale_factor * torch.rand_like(array_total) * noise_level
    else:
        array_total = array_total + torch.rand_like(array_total) * noise_level

    array_total[array_total < 1] = 0
    return torch.round(array_total)

def get_ode_func(K, L, config):
    from .model.ode_block import ODEBlock
    return ODEBlock(
        L=L,
        num_clones=K.shape[0],
        num_pops=K.shape[1],
        hidden_dim=config.hidden_dim, 
        activation=config.activation, 
        K_type=config.K_type,
    )

def simulation(
    K, # (num_clones, num_populations, num_populations)
    L,
    config,
    t_simu=torch.tensor([0.0, 1.0, 2.0, 3.0]), 
    y0=None,
    noise_level=1e-1,
    scale=True,
    raw=False
):
    '''
    Given rates, generate data at raw scale by adding noise. 
    Potential variables to vary: 
        1. Noise level
        2. Number of time points 
        3. Dynamic shapes (up & down) 
        4. (e.g., estimated from seed data)
    '''

    ode_func = get_ode_func(K, L, config)

    ode_func.K1 = Parameter(torch.sqrt(K * ode_func.K1_mask), requires_grad=True)
    ode_func.K2 = Parameter(torch.diagonal(K, dim1=-2, dim2=-1), requires_grad=True)

    array_total = get_array_total(ode_func, y0, t_simu, noise_level, scale, raw=raw)
    return array_total.to(config.gpu)

def simulation_stepK(
    dim_K, # (num_clones, num_populations, num_populations)
    L, # (num_populations, num_populations)
    config,
    t_simu=torch.tensor([0.0, 1.0, 2.0, 3.0]), 
    y0=None,
    noise_level=1e-1
):
    assert L.shape == dim_K[1:]
    assert y0.shape == dim_K[:-1]

    K = torch.normal(0, 0.25, size=dim_K).to(config.gpu)
    ode_func = get_ode_func(K, config)

    # Proliferation
    ode_func.K1 = Parameter(torch.zeros_like(K), requires_grad=True)
    ode_func.K2 = Parameter(torch.diagonal(torch.abs(K), dim1=-2, dim2=-1), requires_grad=True)

    array_proli = odeint(ode_func, y0, t_simu[[0, 1]], rtol=1e-5, method='dopri5', options=dict(dtype=torch.float32))

    # Differentiation
    ode_func.K1 = Parameter(torch.sqrt(torch.abs(K)) * L.unsqueeze(0), requires_grad=True)
    ode_func.K2 = Parameter(torch.diagonal(K, dim1=-2, dim2=-1), requires_grad=True)

    array_diffe = odeint(ode_func, array_proli[1], t_simu[[1, 2, 3]], rtol=1e-5, method='dopri5', options=dict(dtype=torch.float32))
    
    # Concatenate
    array_total = torch.concat([array_proli, array_diffe[1:]])
    array_total[array_total < 1] = 0
    array_total = torch.round(array_total)

    return array_total.to(config.gpu), ode_func.K1, ode_func.K2

def simulation_dynaK(
    K1_encode,
    K1_decode,
    K2_encode,
    K2_decode,
    L,
    config,
    t_simu=torch.tensor([0.0, 1.0, 2.0, 3.0]), 
    y0=None,
    noise_level=1e-1
):
    K = torch.normal(0, 0.25, size=(K2_encode.shape[0], K2_encode.shape[1], K2_encode.shape[1])).to(config.gpu)
    ode_func = get_ode_func(K, L, config)

    ode_func.K1_encode = Parameter(K1_encode, requires_grad=True)
    ode_func.K1_decode = Parameter(K1_decode, requires_grad=True)
    ode_func.K2_encode = Parameter(K2_encode, requires_grad=True)
    ode_func.K2_decode = Parameter(K2_decode, requires_grad=True)

    array_total = get_array_total(ode_func, y0, t_simu, noise_level, raw=True)
    return array_total.to(config.gpu)