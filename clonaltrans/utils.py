import torch
import numpy as np
import time
import inspect
from scipy import interpolate
from torchdiffeq import odeint
from torch import nn
from scipy.optimize import minimize_scalar

GaussianNLL = nn.GaussianNLLLoss(reduction='mean')

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

def input_data_form(N, input_form='log', atol=0.0, exponent=1/4):
    assert input_form in ['log', 'raw', 'shrink', 'root']

    if input_form == 'log':
        return torch.log(N + atol)
    if input_form == 'raw':
        return N
    if input_form == 'shrink':
        return N / 1e5
    if input_form == 'root':
        return torch.pow(N + atol, exponent=exponent)

def val_thres(K, thres=6.):
    remain = np.stack(np.where(K > thres))
    print (f'# of entries where transition rates > {thres}: {remain.shape[1]}')
    print (remain, '\n')

def validate_K(model):
    K = (model.get_matrix_K(eval=True)).detach().cpu().numpy()

    val_thres(K, 10.)
    val_thres(K, 6.)
    val_thres(K, 4.)

    non_dia_mask = model.L.unsqueeze(0).cpu().numpy()
    non_diagonal = K * non_dia_mask
    print (f'# of non-diagonal entries (in topology L) < 0 among all clones: {np.sum(non_diagonal < 0)}')
    print (np.stack(np.where(non_diagonal < 0)), '\n')

    oppo_mask = model.oppo_L.cpu().numpy()
    oppo = K * oppo_mask
    oppo = oppo[np.where(oppo != 0)]
    print ('All other entries not in topology graph L should be as close to 0 as possible, ideally strictly equals to 0.')
    print (f'# of entries: {np.sum(oppo_mask)}')
    print (f'Max: {np.max(oppo):.6f}, Median: {np.median(oppo):.6f}, Min: {np.min(oppo):.6f}')