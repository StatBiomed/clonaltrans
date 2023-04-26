import pandas as pd
import os
import torch
import numpy as np
import time
import inspect

def get_topo_obs(
    data_dir, 
    fill_diagonal: bool = False, 
    simulation: bool = False,
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
        init_con = pd.read_csv('./data/initial_condition.csv', index_col=0).astype(np.float32)
        day_zero = np.zeros((array_ori.shape[1], array_ori.shape[2]))
        day_zero[:, 0] = init_con['leiden'].values
        array_ori = torch.concatenate((torch.tensor(day_zero, dtype=torch.float32).unsqueeze(0), array_ori), axis=0)
        print (f'Day 0 has been added. Input data shape: {array_ori.shape}')

    # generate background cells
    background = torch.sum(array_ori, axis=1).unsqueeze(1)
    array_total = torch.concatenate((array_ori, background), axis=1)
    print (f'Background reference cells generated. Input data shape: {array_total.shape}')

    if False:
        array_total = simulation_data(array_total)
        print (f'Simulation data generated. Final input shape {array_total.shape}')

    return torch.tensor(paga.values, dtype=torch.float32, device=device), array_total.to(device)

def simulation_data(arr):
    shape = arr[0].shape
    arr[2] = arr[1] * 20 + torch.normal(torch.zeros(shape), torch.ones(shape)) * 20
    arr[2][arr[2] < 0] = 0

    arr[3] = arr[2] / 5 + 50 + torch.normal(torch.zeros(shape), torch.ones(shape)) * 10
    arr[3][arr[3] < 0] = 0

    return arr

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.autograd.set_detect_anomaly(True)
    torch.set_printoptions(sci_mode=False)

def init_config_summary(config=None):
    from .config import Configuration
    if config == None:
        print (f'Model configuration file not specified. Default settings will be used.')
        config = Configuration()

    print ('------> Manully Specified Parameters <------')
    config_ref = Configuration()
    dict_input, dict_ref = vars(config), vars(config_ref)

    para_used = []
    for parameter in dict_ref:
        if dict_input[parameter] != dict_ref[parameter]:
            print (parameter, dict_input[parameter], sep=f':\t')
            para_used.append(parameter)

    print ('------> Model Configuration Settings <------')
    for parameter in list(vars(config).keys()):
        if parameter not in para_used:
            print (parameter, dict_ref[parameter], sep=f':\t')
    
    print ('--------------------------------------------')
    print ('')
    return config

def timeit(func, epoch, writer):
    def wrapper(*args, **kwargs):
        start_time = time.monotonic()
        result = func(*args, **kwargs)
        end_time = time.monotonic()

        # Use inspect to get the line number of the decorated function call
        # line_number = inspect.currentframe().f_back.f_lineno
        # print(f"Line {line_number} in {func.__name__} took {end_time - start_time:.6f} seconds")
        
        tb_scalar(
            [f'Time/{func.__name__}'],
            [np.round(end_time - start_time, 3)],
            epoch, writer
        )

        return result
    return wrapper

def tb_scalar(var_names, var_lists, iter, writer):
    for idx, variable in enumerate(var_lists):
        writer.add_scalar(var_names[idx], variable, iter)

def pbar_descrip(var_names, var_lists):
    res = ''
    for idx, variable in enumerate(var_lists):
        res += f'{var_names[idx]} {variable:.3f}, '
    
    return res

def pbar_tb_description(var_names, var_lists, iter, writer):
    res = ''
    for idx, variable in enumerate(var_lists):
        res += f'{var_names[idx]} {variable:.3f}, '
        writer.add_scalar(var_names[idx], variable, iter)
    
    return res

def input_data_form(N, input_form='log', atol=1e-1, exponent=1/4):
    assert input_form in ['log', 'raw', 'shrink', 'root']

    if input_form == 'log':
        return torch.log(N + atol)
    if input_form == 'raw':
        return N
    if input_form == 'shrink':
        return N / 1e5
    if input_form == 'root':
        return torch.pow(N, exponent=exponent)

def val_thres(K, thres=6.):
    remain = np.stack(np.where(K > thres))
    print (f'# of entries where transition rates > {thres}: {remain.shape[1]}')
    print (remain, '\n')

def validate_K(model):
    K = (model.get_matrix_K(eval=True) * 4).detach().cpu().numpy()

    val_thres(K, 6.)
    val_thres(K, 4.)

    non_dia_mask = model.L.unsqueeze(0).cpu().numpy()
    non_diagonal = K * non_dia_mask
    print (f'# of non-diagonal entries < 0 among all clones: {np.sum(non_diagonal < 0)}')
    print (np.stack(np.where(non_diagonal < 0)), '\n')

    oppo_mask = model.oppo_L.cpu().numpy()
    oppo = K * oppo_mask
    oppo = oppo[np.where(oppo != 0)]
    print ('All other entries not in topology graph L should be as close to 0 as possible, ideally strictly equals to 0.')
    print (f'# of entries: {np.sum(oppo_mask)}')
    print (f'Max: {np.max(oppo):.6f}, Median: {np.median(oppo):.6f}, Min: {np.min(oppo):.6f}')