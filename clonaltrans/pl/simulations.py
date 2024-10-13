from scipy.stats import spearmanr, pearsonr
import itertools
import torch
import numpy as np
import os
from natsort import natsorted
from utils import get_post_masks
from tqdm import tqdm

def get_simu_models(dir_simu):
    simu_models = []

    for model in natsorted(os.listdir(dir_simu)):
        if model.endswith('.pt'):
            model_path = os.path.join(dir_simu, model)
        else:
            model_path = os.path.join(dir_simu, model, 'model_last.pt')
        simu_models.append(torch.load(model_path, map_location='cuda:0'))
    
    return simu_models

def get_simu_Ks(simu_models, tpoint):
    simu_Ks = []

    for model in simu_models:
        K_type = model.config['arch']['args']['K_type']
        K = model.get_matrix_K(K_type=K_type, eval=True, tpoint=tpoint).detach().cpu().numpy()
        
        masks = get_post_masks(model, tpoint)
        K[masks[0], masks[1], :] = 0
        simu_Ks.append(K)
    
    return simu_Ks

def get_correlations_pop(N_simu, N_pred):
    corr = [[] for i in range(len(N_pred))]

    for i, t, c in itertools.product(
        range(0, len(N_pred)), 
        range(0, N_simu.shape[0] - 1), 
        range(0, N_simu.shape[1])
    ):
        stats, _ = pearsonr(N_simu[t + 1, c].flatten(), N_pred[i][t + 1, c].flatten())
        corr[i].append(stats)
    
    return corr

def get_correlations_all_log(N_simu, N_pred):
    x, y, stats = [], [], []
    for i in range(len(N_pred)):
        x_item = np.log(np.round(N_simu[1:].flatten()) + 1)
        y_item = np.log(np.round(N_pred[i][1:].flatten()) + 1)

        x.append(x_item)
        y.append(y_item)
        stats.append(pearsonr(x_item, y_item)[0])
    
    return x, y, stats

def get_validation_l1(N_simu, N_pred):
    losses = []

    for i in range(0, len(N_pred)):
        losses.append(np.abs(N_simu[1:] - N_pred[i][1:]))
        losses[-1] = (1- np.mean(losses[-1], axis=(1, 2)) / np.mean(N_simu[1:], axis=(1, 2))) * 100
    
    return losses

def get_corr_losses(dir_simu, N_simu, valid_times=[0.0, 1.0]):
    simu_models = get_simu_models(dir_simu)
    t_pred = torch.tensor(valid_times).to('cpu')

    N_pred = []
    for model in simu_models:
        N_pred.append(model.eval_model(t_pred).detach().cpu().numpy())
    
    corr = get_correlations_pop(N_simu.detach().cpu().numpy(), N_pred)
    losses = get_validation_l1(N_simu.detach().cpu().numpy(), N_pred)
    corr_all_log = get_correlations_all_log(N_simu.detach().cpu().numpy(), N_pred)

    return corr, losses, corr_all_log

def get_error_rate(dir_simu, model_ori):
    simu_models = get_simu_models(dir_simu)
    simu_models.append(model_ori)

    x = torch.linspace(0, int(model_ori.t_observed[-1].item()), int(model_ori.t_observed[-1].item() + 1)).to('cpu')
    mae_total, mean_total = [], []

    for time in tqdm(x):
        simu_Ks = get_simu_Ks(simu_models, time)

        for idx, Ks in enumerate(simu_Ks):
            simu_Ks[idx] = Ks[np.where(np.broadcast_to(model_ori.used_L, simu_Ks[-1].shape) == 1)].flatten()

        mae = [np.abs(Ks - simu_Ks[-1]) for Ks in simu_Ks][:-1]
        mean = [np.mean(item) for item in mae]
    
        mae_total.append(mae)
        mean_total.append(mean)
    
    return mae_total, mean_total