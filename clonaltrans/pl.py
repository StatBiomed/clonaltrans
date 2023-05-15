from scipy.stats import spearmanr
from torch import nn
import matplotlib.pyplot as plt
import math
import seaborn as sns
import numpy as np
import pandas as pd
import torch

MSE = nn.MSELoss(reduction='mean')
SmoothL1 = nn.SmoothL1Loss(reduction='mean', beta=1.0)

def get_subplot_dimensions(
    num_plots, 
    max_cols: int = 5, 
    fig_width_per_col: int = 5, 
    fig_height_per_row: int = 2
):
    cols = min(num_plots, max_cols)
    rows = math.ceil(num_plots / cols)

    fig_width = fig_width_per_col * cols
    fig_height = fig_height_per_row * rows
    return rows, cols, (fig_width, fig_height)

def mse_corr(
    observations, 
    predictions, 
    t_observed, 
    size=20, 
    hue=None, 
    palette=None, 
    save=False
):
    num_t = observations.shape[0]
    print (f'There are {num_t} observed timepoints except the inital time.')

    from .pl import get_subplot_dimensions
    rows, cols, figsize = get_subplot_dimensions(num_t, fig_height_per_row=4)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    anno = pd.read_csv('./data/annotations.csv')
    if hue == 'pop':
        pop = np.broadcast_to(anno['populations'].values, (observations[0].shape)).flatten()

    for n in range(num_t):
        loss = SmoothL1(observations[n], predictions[n])
        x = observations[n].cpu().numpy().flatten()
        y = predictions[n].detach().cpu().numpy().flatten()
        spear = spearmanr(x, y)[0]
        
        ax_loc = axes[n % cols][n // cols] if rows > 1 else axes[n]
        sns.scatterplot(
            x=x, y=y, s=size, ax=ax_loc,
            hue=hue, palette=palette
        )
        ax_loc.set_title(f'Time {t_observed[n]} Loss {loss.item():.3f} Corr {spear:.3f}')
        ax_loc.set_xlabel(f'Observations')
        ax_loc.set_ylabel(f'Predictions')

    if save:
        plt.savefig(f'./figs/{save}.png', dpi=300, bbox_inches='tight')

def plot_gvi(data, axes, row, col, t_axis, label, color):
    if data is not None:
        if len(data) > 10:
            axes[row][col].plot(
                t_axis, 
                data[:, row, col], 
                label=label, 
                color=color,
            )
        else:
            axes[row][col].plot(
                t_axis, 
                data[:, row, col], 
                label=label, 
                color=color,
                marker='o',
                linestyle='',
            )

def grid_visual_interpolate(
    data_values: list = [any, any, None],
    data_names: list = ['Observations', 'Predictions', None],
    data_tpoint: list = [any, None, None],
    variance: bool = False,
    raw_data: bool = False,
    save: bool = False
):
    '''
    data_values[0] must be true observational data tensor
    data_values[1] must be the predicted data
    '''
        
    fig, axes = plt.subplots(data_values[0].shape[1], data_values[0].shape[2], figsize=(33, 20), sharex=True)
    
    obs = data_values[0].cpu().numpy()
    pred = data_values[1].detach().cpu().numpy()
    pred2 = data_values[2].detach().cpu().numpy() if data_values[2] != None else None

    t_obs = data_tpoint[0].cpu().numpy()
    t_pred = data_tpoint[1].cpu().numpy()
    t_pred2 = data_tpoint[2].cpu().numpy() if data_tpoint[2] != None else None

    anno = pd.read_csv('./data/annotations.csv')

    if variance is not False:
        lb = pred - np.sqrt(variance)
        ub = pred + np.sqrt(variance)

        lb[lb < 0] = 0
        ub[ub < 0] = 0

    if raw_data:
        obs = np.power(obs, 4)
        pred = np.power(pred, 4)
        if variance is not False:
            lb, ub = np.power(lb, 4), np.power(ub, 4)

    for row in range(data_values[0].shape[1]):
        for col in range(data_values[0].shape[2]):
            if variance is not False:
                axes[row][col].fill_between(
                    t_pred,
                    lb[:, row, col],
                    ub[:, row, col],
                    label='1 std error',
                    color='lightskyblue',
                    alpha=0.5
                )

            plot_gvi(pred2, axes, row, col, t_pred2, data_names[2], 'skyblue')
            plot_gvi(pred, axes, row, col, t_pred, data_names[1], 'lightcoral')
            plot_gvi(obs, axes, row, col, t_obs, data_names[0], '#2C6975')
    
            axes[0][col].set_title(anno['populations'][col])
            axes[row][0].set_ylabel(anno['clones'][row])
            axes[row][col].set_xticks(t_obs, labels=t_obs.astype(int), rotation=45)

    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.5)
    handles, labels = axes[row][col].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), fontsize='x-large')

    if save:
        plt.savefig(f'./figs/{save}.png', dpi=300, bbox_inches='tight')

def clone_specific_K(model, index_clone=0):
    K = (model.get_matrix_K(eval=True)).detach().cpu().numpy()
    anno = pd.read_csv('./data/annotations.csv')

    transition_K = pd.DataFrame(
        index=anno['populations'].values, 
        columns=anno['populations'].values, 
        data=K[index_clone]
    )
    fig, axes = plt.subplots(figsize=(12, 6))
    sns.heatmap(transition_K, annot=True, linewidths=.5, cmap='coolwarm', ax=axes)
    plt.title(f'Transition Matrix for Clone {index_clone}')

def diff_K_between_clones(model, index_pop=0, direction='outbound'):
    K = (model.get_matrix_K(eval=True)).detach().cpu().numpy()
    anno = pd.read_csv('./data/annotations.csv')
    pop_name = anno['populations'].values[index_pop]

    if direction == 'outbound':
        transition_K = pd.DataFrame(
            index=anno['clones'].values,
            columns=pop_name + ' -> ' + anno['populations'].values,
            data=K[:, index_pop, :]
        ).T
    if direction == 'inbound':
        transition_K = pd.DataFrame(
            index=anno['clones'].values,
            columns=anno['populations'].values + ' -> ' + pop_name,
            data=K[:, :, index_pop]
        ).T 

    fig, axes = plt.subplots(figsize=(12, 6))
    sns.heatmap(transition_K, annot=True, linewidths=.5, cmap='coolwarm', ax=axes)
    plt.xticks(rotation=0)
    plt.title(f'Difference in transition rates between clones for {pop_name} ({direction})')

def rates_notin_paga(model):
    K = (model.get_matrix_K(eval=True)).detach().cpu().numpy()
    oppo = K * model.oppo_L.cpu().numpy()
    nonzeros = oppo[np.where(oppo != 0)]

    print ('All other entries not in topology graph L should be as close to 0 as possible, \nideally strictly equals to 0.')
    print (f'# of entries: {np.sum(model.oppo_L.cpu().numpy())}, # of zeros: {np.sum(model.oppo_L.cpu().numpy()) - len(nonzeros)}, Details of nonzeros: ')
    print (f'Max: {np.max(nonzeros):.6f}, Median: {np.median(nonzeros):.6f}, Min: {np.min(nonzeros):.6f}')

    sns.histplot(nonzeros, bins=10)
    plt.title(f'Distribution of non-zero rates not in PAGA')

def rates_in_paga(model):
    K = (model.get_matrix_K(eval=True)).detach().cpu().numpy()
    used_K = K[np.where(model.used_L.cpu().numpy())]
    sns.histplot(used_K, bins=30)
    plt.title(f'Distribution of rates in PAGA')

def rates_diagonal(model):
    K = (model.get_matrix_K(eval=True))
    diag = torch.diagonal(K, dim1=-2, dim2=-1).detach().cpu().numpy()
    anno = pd.read_csv('./data/annotations.csv')

    transition_K = pd.DataFrame(
        index=anno['clones'].values, 
        columns=anno['populations'].values, 
        data=diag
    ).T

    fig, axes = plt.subplots(figsize=(12, 6))
    sns.heatmap(transition_K, annot=True, linewidths=.5, cmap='coolwarm', ax=axes)
    plt.title(f'Diagonal of transition rates (Proliferation & Apoptosis)')