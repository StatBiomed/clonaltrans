from torch import nn
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns
from .base import get_subplot_dimensions, transit_K
import numpy as np 
from utils import get_K_total, get_post_masks
from scipy.stats import pearsonr
import torch
import pandas as pd
import os
from itertools import combinations, product

SmoothL1 = nn.SmoothL1Loss(reduction='mean', beta=1.0)

def mse_corr(model, save=False):
    observations = model.N
    predictions = model.eval_model(model.t_observed)

    if 'sample_N' not in dir(model):
        sample_N = torch.ones(observations.shape)
    else:
        sample_N = model.sample_N

    num_t = observations.shape[0]
    rows, cols, figsize = get_subplot_dimensions(num_t - 1, fig_height_per_row=4)
    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] + 5, figsize[1]))

    for n in range(1, num_t):
        x = (observations[n].cpu() * sample_N[n].cpu()).detach()
        y = (predictions[n].cpu() * sample_N[n].cpu()).detach()

        loss = SmoothL1(x, y)
        spear = spearmanr(x.numpy().flatten(), y.numpy().flatten())[0]
        title = f'Time {model.t_observed[n]} Loss {loss.item():.3f} Corr {spear:.3f} '
        
        ax_loc = axes[(n - 1) // cols][(n - 1) % cols] if rows > 1 else axes[n - 1]
        ax_loc.plot([x.min(), x.max()], [x.min(), x.max()], linestyle="--", color="grey")
        sns.scatterplot(x=x.numpy().flatten().astype(int), y=y.numpy().flatten().astype(int), s=20, ax=ax_loc)

        ax_loc.set_title(title)
        ax_loc.set_xlabel(f'Observations')
        ax_loc.set_ylabel(f'Predictions')
        # ax_loc.ticklabel_format(axis='both', style='sci', scilimits=(0, 4))
        ax_loc.set_yscale('log')
        ax_loc.set_xscale('log')

    if save:
        plt.savefig(f'./figs/{save}.png', dpi=300, bbox_inches='tight', transparent=False, facecolor='white')

def clone_specific_K(model, index_clone=0, tpoint=1.0, save=False):
    K_type = model.config['arch']['args']['K_type']
    K = model.get_matrix_K(K_type=K_type, eval=True, tpoint=tpoint).detach().cpu().numpy()
    print ('---> Negative values were cut to the opposite of positive values for visualization purpose. <---')

    masks = get_post_masks(model, tpoint)
    K[masks[0], masks[1], :] = 0

    df = transit_K(model, K[index_clone])
    _, axes = plt.subplots(1, 1, figsize=(8, 6))
    _ = sns.heatmap(df, linewidths=.5, cmap='coolwarm', ax=axes, vmin=-np.max(df.values), vmax=np.max(df.values))
    
    index_clone = index_clone if index_clone != -1 else 'BG'
    title = f'Transition rates for Clone {index_clone} | Day {np.round(tpoint, 1)}' \
        if K_type == 'dynamic' else f'Transition rates for clone {index_clone}'
    plt.title(title, fontsize=13)

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

def rates_in_paga(model, save=False):
    K_total = get_K_total(model)[:, :-1, :, :]
    used_L = model.used_L.detach().cpu().numpy()

    ax = sns.histplot(K_total[np.broadcast_to(np.expand_dims(used_L, 0), K_total.shape) != 0].flatten(), bins=50)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.ylabel('Count', fontsize=13)
    plt.xlabel(f'Per capita transition rates of {model.N.shape[1] - 1} meta-clones', fontsize=13)
    plt.title(f'Distribution of rates (Day {model.t_observed[0]} ~ Day {model.t_observed[-1]})', fontsize=13)
    plt.text(4, 1000, f'Each meta-clone has:\n{used_L.shape[1]} proliferation & \n{np.sum(used_L != 0) - used_L.shape[1]} differentiation rates', fontsize=13)
    plt.yscale('log')

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

def rates_notin_paga(model, save=False):
    K_total = get_K_total(model)
    oppo_L = model.oppo_L.detach().cpu().numpy()

    not_in = K_total[np.broadcast_to(np.expand_dims(oppo_L, 0), K_total.shape) != 0].flatten()
    ax = sns.histplot(not_in[not_in != 0], bins=50)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.title(f'Rates not in PAGA that are non-zero')

    if save:
        plt.savefig(f'./figs/{save}.png', dpi=300, bbox_inches='tight', transparent=False, facecolor='white')

def rates_diagonal(model, tpoint=1.0, save=False):
    K_type = model.config['arch']['args']['K_type']
    K = model.get_matrix_K(K_type=K_type, eval=True, tpoint=tpoint).detach().cpu().numpy()

    masks = get_post_masks(model, tpoint)
    K[masks[0], masks[1], :] = 0
    K = torch.tensor(K, dtype=torch.float32)

    diag = torch.diagonal(K, dim1=-2, dim2=-1).detach().cpu().numpy()
    print ('---> Negative values were cut to the opposite of positive values for visualization purpose. <---')

    anno = pd.read_csv(os.path.join(model.config['data_loader']['args']['data_dir'], model.config['data_loader']['args']['annots']))
    df = transit_K(model, diag, [item.split(' ')[1] for item in anno['clones'].values[:K.shape[0]]]).T

    fig, axes = plt.subplots(figsize=(10, 4))
    sns.heatmap(df, linewidths=.5, cmap='coolwarm', ax=axes, vmin=-np.max(df.values), vmax=np.max(df.values))
    plt.title(f'Diagonal of transition rates (Proliferation & Apoptosis)')

    if save:
        plt.savefig(f'./figs/{save}.png', dpi=300, bbox_inches='tight', transparent=False, facecolor='white')

def compare_with_bg(model, save=False):
    K_total = get_K_total(model)
    x = np.mean(K_total[:, :-1, :, :], axis=1).flatten()
    y = K_total[:, -1, :, :].flatten()

    corr, p_value = pearsonr(x, y)
    ax = sns.scatterplot(x=y, y=x, s=20, c='#2c6aab')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot([x.min(), x.max()], [x.min(), x.max()], linestyle="--", color="grey")
    plt.xlabel(f'All cells (incl. filtered at pre-processing)', fontsize=13)
    plt.ylabel(f'Mean of {K_total.shape[1] - 1} meta-clones', fontsize=13)
    plt.text(-0.5, 1, f'$Pearson \; r = {corr:.3f}$', fontsize=13)
    plt.title(f'Distribution of rates (Day {model.t_observed[0]} ~ Day {model.t_observed[-1]})', fontsize=13)

    if save is not False:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

def clone_rates_diff_plot(
    adjusted_p_values,
    ref_model,
    save=False
):
    anno = pd.read_csv(os.path.join(
        ref_model.config['data_loader']['args']['data_dir'], 
        ref_model.config['data_loader']['args']['annots']
    ))
    graph = pd.read_csv(os.path.join(
        ref_model.config['data_loader']['args']['data_dir'], 
        ref_model.config['data_loader']['args']['graphs']
    ), index_col=0)
    np.fill_diagonal(graph.values, 1)
    num_clones = ref_model.N.shape[1]
    
    cols = []
    for pop1, pop2 in product(range(graph.shape[0]), range(graph.shape[1])):
        if graph.values[pop1][pop2] != 0:
            cols.append('{} -> {}'.format(anno['populations'].values[pop1], anno['populations'].values[pop2]))

    index = []
    for (c1, c2) in combinations(range(num_clones), 2):
        if c1 == num_clones - 1:
            c1 = 'BG'
        if c2 == num_clones - 1:
            c2 = 'BG'
        index.append(f'Clone {c1} / {c2}')

    fig, axes = plt.subplots(figsize=(50, 12))
    adjusted_p_values[adjusted_p_values > 0.05] = np.nan

    df = pd.DataFrame(data=-np.log10(adjusted_p_values), index=cols, columns=index)
    ax = sns.heatmap(df, annot=False, linewidths=.5, cmap='viridis')
    plt.title('$-log_{10}$ p-value across meta-clones & populations', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

def get_rates_avg(model, save: bool = False):
    K_total = np.mean(get_K_total(model), axis=0)
    num_clones = model.N.shape[1]

    prol, diff = [], []
    for i in range(num_clones):
        prol.append(np.mean(K_total[i].diagonal()[np.abs(K_total[i].diagonal()) > 0]))

        np.fill_diagonal(K_total[i], 0)
        diff.append(np.mean(K_total[i][np.abs(K_total[i]) > 0].flatten()))
    
    df = pd.DataFrame({'Proliferation': prol, 'Differeation': diff})
    df.index = [f'Clone {i}' if i != num_clones - 1 else 'Clone BG' for i in range(num_clones)]

    ax = df.plot(kind='bar', stacked=False, figsize=(8,4), width=0.8, colormap='tab20')
    plt.legend(bbox_to_anchor=(0.25, 1), frameon=False)
    plt.xticks(rotation=90, fontsize=10)
    plt.ylabel('Average rates among populations', fontsize=12)

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')