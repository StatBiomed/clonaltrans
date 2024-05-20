import torch
import numpy as np
import time
from pathlib import Path
import json
from collections import OrderedDict
from functools import wraps
import os 
import pandas as pd
from itertools import combinations, product
import scipy.stats as stats
import statsmodels.stats.multitest as smm
from tqdm import tqdm

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def time_func(func):
    @wraps(func)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)

        t2 = time.time()
        print(f"@time_func: {func.__name__} took {t2 - t1: .5f} s")

        return result
    return measure_time

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.autograd.set_detect_anomaly(True)
    torch.set_printoptions(sci_mode=False)

def pbar_descrip(var_names, var_lists):
    res = ''
    for idx, variable in enumerate(var_lists):
        res += f'{var_names[idx]} {variable:.3f}, '
    
    return res

def pbar_tb_description(var_names, var_lists, iter, writer):
    res = ''
    for idx, variable in enumerate(var_lists):
        if type(variable) != str:
            res += f'{var_names[idx]} {variable:.3f}, '
        else:
            res += f'{var_names[idx]} {variable}, '

        if writer is not None:
            writer.add_scalar(var_names[idx], variable, iter)
    
    return res

def get_post_masks(model, tpoint):
    predictions = model.eval_model(torch.tensor([0.0, max(tpoint, 0.001)]).to(model.config['system']['gpu_id']))[1]
    masks = np.where(predictions.detach().cpu().numpy() < 0.5)
    return masks

def get_K_total(model, tpoints=None):
    K_total = []
    K_type = model.config['arch']['args']['K_type']
    gpu = model.config['system']['gpu_id']

    if tpoints is None:
        x = torch.linspace(0, int(model.t_observed[-1].item()), int(model.t_observed[-1].item() + 1)).to(gpu)
    else:
        x = tpoints

    for i in range(len(x)):
        masks = get_post_masks(model, x[i])
        K = model.get_matrix_K(K_type=K_type, eval=True, tpoint=x[i]).detach().cpu().numpy()
        K[masks[0], masks[1], :] = 0
        K_total.append(K)
    
    return np.stack(K_total) # (num_time_points, num_clones, num_pops, num_pops)

def get_boots_K_total(model_list, ref_model=None, K_type='const', tpoint=1.0):
    total_K = []

    for model in model_list:
        masks = get_post_masks(model, tpoint)
        K = model.get_matrix_K(K_type=K_type, eval=True, tpoint=tpoint).detach().cpu().numpy()
        K[masks[0], masks[1], :] = 0
        total_K.append(K)

    masks = get_post_masks(ref_model, tpoint)
    ref_K = ref_model.get_matrix_K(K_type=K_type, eval=True, tpoint=tpoint).detach().cpu().numpy()
    ref_K[masks[0], masks[1], :] = 0

    total_K = np.stack(total_K)
    return total_K, ref_K # (b, c, p, p) (c, p, p)

def clone_rates_diff_test(ref_model, total_K, correct_method='fdr_bh'):
    paga = pd.read_csv(os.path.join(
        ref_model.config['data_loader']['args']['data_dir'], 
        ref_model.config['data_loader']['args']['graphs']), 
        index_col=0).astype(np.int32).values
    np.fill_diagonal(paga, 1)

    num_clone, num_pop = ref_model.N.shape[1], ref_model.N.shape[2]
    stats_tests = np.zeros((int(num_clone * (num_clone - 1) / 2), num_pop, num_pop))
    stats_tests[stats_tests == 0] = 'nan'

    fold_change = np.zeros((int(num_clone * (num_clone - 1) / 2), num_pop, num_pop))
    fold_change[fold_change == 0] = 'nan'

    count = 0
    for (c1, c2) in combinations(range(num_clone), 2):
        for pop1, pop2 in product(range(num_pop), range(num_pop)):

            if paga[pop1, pop2] == 1:
                K_c1 = total_K[:, c1, pop1, pop2]
                K_c2 = total_K[:, c2, pop1, pop2]

                fold_change[count, pop1, pop2] = np.mean(np.abs(K_c1 - K_c2))

                K_c1 = K_c1[(K_c1 > np.percentile(K_c1, 2.5)) & (K_c1 < np.percentile(K_c1, 97.5))]
                K_c2 = K_c2[(K_c2 > np.percentile(K_c2, 2.5)) & (K_c2 < np.percentile(K_c2, 97.5))]

                try:
                    _, shapiro_p_c1 = stats.shapiro(K_c1)
                    _, shapiro_p_c2 = stats.shapiro(K_c2)

                    if shapiro_p_c1 > 0.05 and shapiro_p_c2 > 0.05:
                        _, paired_p = stats.ttest_ind(K_c1, K_c2) 
                        stats_tests[count, pop1, pop2] = paired_p
                        
                    else:
                        _, wilcox_p = stats.mannwhitneyu(K_c1, K_c2)
                        stats_tests[count, pop1, pop2] = wilcox_p
                except:
                    stats_tests[count, pop1, pop2] = 1
        count += 1
    
    stats_correct = stats_tests.reshape((stats_tests.shape[0], stats_tests.shape[1] * stats_tests.shape[1]))
    nan_cols = np.isnan(stats_correct).any(axis=0)
    stats_correct = stats_correct[:, ~nan_cols]

    fold_correct = fold_change.reshape((fold_change.shape[0], fold_change.shape[1] * fold_change.shape[1]))
    nan_cols = np.isnan(fold_correct).any(axis=0)
    fold_correct = fold_correct[:, ~nan_cols]

    adjusted_p_values = []
    for i in range(stats_correct.shape[1]):
        adjusted_p_values.append(smm.multipletests(stats_correct[:, i], method=correct_method)[1])

    return stats_tests, np.stack(adjusted_p_values), fold_correct.T

def get_boots_K_total_with_time(model_list, ref_model=None, tpoints=None):
    K_total = []
    ref_K_total = []
    N_total = []
    K_type = ref_model.config['arch']['args']['K_type']

    if tpoints is None:
        x = torch.linspace(0, int(ref_model.t_observed[-1].item()), int(ref_model.t_observed[-1].item() + 1)).to('cpu')
    else:
        x = tpoints
    
    pbar = tqdm(range(len(x)))
    for i in pbar:
        total_K, ref_K = get_boots_K_total(model_list, ref_model=ref_model, K_type=K_type, tpoint=x[i])

        K_total.append(total_K) 
        ref_K_total.append(ref_K) 

        N = ref_model.eval_model(torch.tensor([0.0, max(x[i], 0.001)]))[1]
        N_total.append(N)
    
    K_total = np.stack(K_total) #* (time, num_bootstraps, clones, pops, pops)
    N_total = np.stack(N_total) #* (time, clones, pops)
    ref_K_total = np.stack(ref_K_total) #* (time, clones, pops, pops)
    
    return K_total, ref_K_total, N_total