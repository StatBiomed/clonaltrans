import torch
from torchdiffeq import odeint
from torch.nn.parameter import Parameter
from model.ode_block import ODESolver

def init_solver(L, K, config):
    return ODESolver(
        L=L,
        num_clones=K.shape[0],
        num_pops=K.shape[1],
        hidden_dim=config['arch']['args']['hidden_dim'], 
        activation=config['arch']['args']['activation'], 
        K_type=config['arch']['args']['K_type'],
        adjoint=config['user_trainer']['adjoint'],
        clipping=config['arch']['args']['clipping']
    ).to(config['system']['gpu_id'])

def simulation_const(
    K, # (num_clones, num_populations, num_populations)
    L,
    config,
    scaling_factor=None,
    t_simu=torch.tensor([0.0, 1.0, 2.0, 3.0]), 
    y=None,
):
    assert config['arch']['args']['K_type'] == 'const', 'const config file shoule be provided'
    K = K.to(config['system']['gpu_id'])
    y = y.to(config['system']['gpu_id'])
    
    solver = init_solver(L, K, config)

    solver.block.K1 = Parameter(torch.sqrt(K * solver.block.K1_mask), requires_grad=True)
    solver.block.K2 = Parameter(torch.diagonal(K, dim1=-2, dim2=-1), requires_grad=True)

    N = solver(y, t_simu)

    if scaling_factor != None:
        for time in range(1, N.shape[0]):
            N[time] += (torch.normal(0, 2, size=(N.shape[1], N.shape[2])) * scaling_factor[time]).to(N.get_device())

        N[torch.where(N < 0)] = 0
    return N

def simulation_dyna(
    K1_encode,
    K1_decode,
    K2_encode,
    K2_decode,
    L,
    config,
    t_simu=torch.tensor([0.0, 1.0, 2.0, 3.0]), 
    y=None
):
    assert config['arch']['args']['K_type'] == 'dynamic', 'dynamic config file shoule be provided'

    K1_encode = K1_encode.to(config['system']['gpu_id'])
    K1_decode = K1_decode.to(config['system']['gpu_id'])
    K2_encode = K2_encode.to(config['system']['gpu_id'])
    K2_decode = K2_decode.to(config['system']['gpu_id'])
    y = y.to(config['system']['gpu_id'])
    
    solver = init_solver(L, K1_encode, config)

    solver.block.K1_encode = Parameter(K1_encode, requires_grad=True)
    solver.block.K1_decode = Parameter(K1_decode, requires_grad=True)
    solver.block.K2_encode = Parameter(K2_encode, requires_grad=True)
    solver.block.K2_decode = Parameter(K2_decode, requires_grad=True)

    return solver(y, t_simu)
