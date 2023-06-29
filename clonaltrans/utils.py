import torch
import numpy as np
import time
import inspect
from scipy import interpolate
from torchdiffeq import odeint
from torch import nn
from scipy.optimize import minimize_scalar
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset

GaussianNLL = nn.GaussianNLLLoss(reduction='mean')

class Clonal_Info(Dataset):
    def __init__(
        self,
        N, # (num_timpoints, num_clones, num_populations)
        input_form: str = 'root', 
        exponent: float = 1 / 4,
    ):
        self.input_N = input_data_form(N, input_form, exponent)

    def __len__(self):
        return self.input_N.size(1)

    def __getitem__(self, idx):
        return self.data[idx, :, :], idx

def get_dataloader(N, input_form, exponent):
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        Clonal_Info(N, input_form, exponent),
        batch_size=N.shape[1],
        shuffle=False
    )

    for batch_data, idx_time in train_loader:
        print (batch_data.shape, idx_time)

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

def input_data_form(N, input_form='root', exponent=1/4):
    assert input_form in ['log', 'raw', 'shrink', 'root']

    if input_form == 'log':
        return torch.log(N + 1.0)
    if input_form == 'raw':
        return N
    if input_form == 'shrink':
        return N / 1e5
    if input_form == 'root':
        return torch.pow(N, exponent=exponent)

def val_thres(K, thres=6.):
    remain = np.stack(np.where(np.abs(K) > thres))
    print (f'# of entries where transition rates > {thres}: {remain.shape[1]}')
    print (remain, '\n')

def validate_K(model):
    K = (model.get_matrix_K(eval=True)).detach().cpu().numpy()

    val_thres(K, 10.)
    val_thres(K, 6.)

    non_dia_mask = model.L.unsqueeze(0).cpu().numpy()
    non_diagonal = K * non_dia_mask
    print (f'# of non-diagonal entries (in topology L) < 0 among all clones: {np.sum(non_diagonal < 0)}')
    print (np.stack(np.where(non_diagonal < 0)), '\n')

def var_interp_1d(variance, t_observed, kind='linear'):
    t_obs = t_observed.cpu().numpy()
    
    x = np.linspace(0, 17, 100)
    vars_inter = np.zeros((len(x), variance.shape[1], variance.shape[2]), dtype=np.float32)

    for clone in range(variance.shape[1]):
        for pops in range(variance.shape[2]):
            interp = interpolate.interp1d(t_obs, variance[:, clone, pops], kind=kind)
            vars_inter[:, clone, pops] = interp(x)
    
    vars_inter[vars_inter < 0] = 0
    return vars_inter

class TempModel():
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir

def bootstrap():
    pass