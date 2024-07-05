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
from scipy.cluster import hierarchy
import copy

L1loss = nn.L1Loss(reduction='mean')

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

        loss = L1loss(x, y).numpy() / np.mean(x.numpy())
        spear = spearmanr(x.numpy().flatten(), y.numpy().flatten())[0]
        title = f'Timepoint {model.t_observed[n]}'
        
        ax_loc = axes[(n - 1) // cols][(n - 1) % cols] if rows > 1 else axes[n - 1]
        ax_loc.plot([x.min(), x.max()], [x.min(), x.max()], linestyle="--", color="grey", zorder=0)
        
        sns.scatterplot(x=x.numpy().flatten().astype(int), y=y.numpy().flatten().astype(int), s=25, ax=ax_loc)
        ax_loc.text(0.05, 0.85, f'Avg. Recovery Rate: {(1 - loss.item()) * 100:.2f}%\nSpearman Corr: {spear:.3f}', transform=ax_loc.transAxes, fontsize=15)

        ax_loc.set_title(title, fontsize=15)
        ax_loc.set_xlabel(f'Observations', fontsize=15)
        ax_loc.set_ylabel(f'Predictions', fontsize=15)

        ax_loc.set_xscale('log')
        ax_loc.set_yscale('log')
        ax_loc.tick_params(axis='both', labelsize=15)

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=True)

def clone_specific_K(model, index_clone=0, tpoint=1.0, save=False):
    import matplotlib.ticker as ticker
    K_type = model.config['arch']['args']['K_type']
    K = model.get_matrix_K(K_type=K_type, eval=True, tpoint=tpoint).detach().cpu().numpy()
    print ('---> Negative values were cut to the opposite of positive values for visualization purpose. <---')

    masks = get_post_masks(model, tpoint)
    K[masks[0], masks[1], :] = 0

    df = transit_K(model, K[index_clone])
    _, axes = plt.subplots(1, 1, figsize=(8, 6))
    hp = sns.heatmap(df, linewidths=.5, cmap='coolwarm', ax=axes, vmin=-np.max(df.values), vmax=np.max(df.values))

    # Access the colorbar object
    colorbar = hp.collections[0].colorbar
    colorbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    colorbar.ax.tick_params(labelsize=13)

    index_clone = index_clone if index_clone != -1 else 'BG'
    title = f'Transition rates for Meta-clone {index_clone} | Day {np.round(tpoint, 1)}' \
        if K_type == 'dynamic' else f'Transition rates for clone {index_clone}'
    plt.title(title, fontsize=14)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    if save:
        plt.savefig(f'./{save}.svg', dpi=300, bbox_inches='tight', transparent=True)

def rates_in_paga(model, save=False):
    K_total = get_K_total(model)[:, :-1, :, :]
    used_L = model.used_L.detach().cpu().numpy()

    ax = sns.histplot(K_total[np.broadcast_to(np.expand_dims(used_L, 0), K_total.shape) != 0].flatten(), bins=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig = ax.get_figure()
    fig.set_figwidth(8) 
    fig.set_figheight(6)

    plt.ylabel('Counts', fontsize=20)
    plt.xlabel(f'Per capita transition rates ', fontsize=20)
    plt.title(f'Distributions (Day {model.t_observed[0]} ~ Day {model.t_observed[-1]})', fontsize=20)
    
    plt.text(
        0.45, 
        0.7, 
        f'Each meta-clone has:\n{used_L.shape[1]} proliferation & \n{np.sum(used_L != 0) - used_L.shape[1]} differentiation rates', 
        fontsize=20,
        transform=plt.gca().transAxes
    )
    plt.tick_params(axis='both', labelsize=20)
    plt.yscale('log')

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=True)

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
    plt.xlabel(f'All cells (incl. filtered at pre-processing stage)', fontsize=15)
    plt.ylabel(f'Mean of {K_total.shape[1] - 1} meta-clones', fontsize=15)
    plt.text(0.05, 0.85, f'$Pearson \; Corr = {corr:.3f}$', fontsize=13, transform=plt.gca().transAxes)
    plt.title(f'Comparison of rates (Day {model.t_observed[0]} ~ Day {model.t_observed[-1]})', fontsize=15)
    plt.tick_params(axis='both', labelsize=13)

    if save is not False:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=True)

def clone_rates_diff_plot(
    adjusted_p_values,
    ref_model,
    plt_type='pvalue',
    save=False
):
    assert plt_type in ['pvalues', 'foldchange', 'combined']
    assert len(adjusted_p_values) == 2 if plt_type == 'combined' else 'Please provide both p-values and fold change'
    adjusted_p_values = copy.deepcopy(adjusted_p_values)

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
        index.append(f'{c1} / {c2}')

    if plt_type == 'pvalues':
        adjusted_p_values[adjusted_p_values > 0.01] = np.nan
        adjusted_p_values = -np.log10(adjusted_p_values)
        title = '$-log_{10}$ p-values | Day 3.0'

    if plt_type == 'foldchange':
        adjusted_p_values[adjusted_p_values < 0.1] = np.nan
        title = 'Foldchange of rates | Day 3.0'
    
    if plt_type == 'combined':
        adjusted_p_values[0][adjusted_p_values[0] > 0.01] = np.nan
        adjusted_p_values[0][np.where(adjusted_p_values[1] < 0.1)] = np.nan
        adjusted_p_values = -np.log10(adjusted_p_values[0])
        title = '$-log_{10}$ p-values | Day 4.0'

    df = pd.DataFrame(data=adjusted_p_values, index=cols, columns=index)
    # df = df.filter(like='BG')
    # df = get_clustered_heatmap(df)

    ax = sns.heatmap(df, annot=False, linewidths=.1, cmap='viridis', vmin=0, xticklabels=True, yticklabels=True, cbar=True)
    plt.title(title, fontsize=30, pad=10)

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    fig = ax.get_figure()
    fig.set_figwidth(55) 
    fig.set_figheight(25) 

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=25)

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=True)

def get_rates_avg(model, save: bool = False):
    K_total = np.mean(get_K_total(model), axis=0)
    num_clones = model.N.shape[1]

    prol, diff = [], []
    for i in range(num_clones):
        prol.append(np.mean(K_total[i].diagonal()[np.abs(K_total[i].diagonal()) > 0]))
        # prol.append(np.mean(K_total[i].diagonal()[0]))

        np.fill_diagonal(K_total[i], 0)
        diff.append(np.mean(K_total[i][np.abs(K_total[i]) > 0].flatten()))
        # diff.append(np.mean(K_total[i][0][np.abs(K_total[i][0]) > 0].flatten()))
    
    df = pd.DataFrame({'Proliferation': prol, 'Differentiation': diff})
    df.index = [f'{i}' if i != num_clones - 1 else 'BG' for i in range(num_clones)]

    ax = df.plot(kind='bar', stacked=False, figsize=(8, 3), width=0.8, colormap='tab20')
    plt.legend(bbox_to_anchor=(0.23, 0.75), frameon=False)
    plt.xticks(rotation=0, fontsize=10)
    plt.ylabel('Average rates among populations', fontsize=11)
    plt.title('Comparison of rates among meta-clones', fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=True)

def get_clustered_heatmap(df):
    df = df.fillna(0)

    # Perform hierarchical clustering on rows and columns
    row_linkage = hierarchy.linkage(df.values, method='ward', metric='euclidean')
    col_linkage = hierarchy.linkage(df.values.T, method='ward', metric='euclidean')
    # Reorder rows and columns based on clustering
    row_order = hierarchy.leaves_list(row_linkage)
    col_order = hierarchy.leaves_list(col_linkage)

    df = df.iloc[row_order[::-1], col_order[::-1]]
    df[df == 0] = np.nan
    return df

def get_fate_clones(adata, aggre):
    aggre_clones = dict()

    for clone in aggre.keys():
        aggre_clones[clone] = {}
        adata_clone_obs = adata.obs[adata.obs['meta_clones'] == clone[6:]]

        for pop in aggre[clone].keys():
            aggre_clones[clone][pop] = [list(aggre[clone][pop].keys()), []]
        
            for descendent in list(aggre[clone][pop].keys()):
                adata_des = adata_clone_obs[adata_clone_obs['label_man'] == descendent]
                aggre_clones[clone][pop][1].append(len(np.unique(adata_des['clones'].values)))
            
            aggre_clones[clone][pop] = dict(zip(aggre_clones[clone][pop][0], aggre_clones[clone][pop][1]))

    return aggre_clones