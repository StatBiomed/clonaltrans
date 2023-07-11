from scipy.stats import spearmanr
from torch import nn
import matplotlib.pyplot as plt
import math
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import os
import imageio
from matplotlib.lines import Line2D
from tqdm import tqdm

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
    return rows, cols, (fig_width + 1, fig_height + 2)

def mse_corr(
    observations, 
    predictions, 
    t_observed, 
    save=False,
    sample_N=None,
    verbose=False
):
    num_t = observations.shape[0]
    verbose_output = ''

    if not verbose:
        from .pl import get_subplot_dimensions
        rows, cols, figsize = get_subplot_dimensions(num_t - 1, fig_height_per_row=4)
        fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] + 3, figsize[1] - 2))

    for n in range(1, num_t):
        x = (observations[n].cpu() * sample_N[n].cpu()).detach()
        y = (predictions[n].cpu() * sample_N[n].cpu()).detach()

        loss = SmoothL1(x, y)
        spear = spearmanr(x.numpy().flatten(), y.numpy().flatten())[0]
        title = f'Time {t_observed[n]} Loss {loss.item():.3f} Corr {spear:.3f} '
        verbose_output += title
        
        if not verbose:
            ax_loc = axes[(n - 1) // cols][(n - 1) % cols] if rows > 1 else axes[n - 1]
            sns.scatterplot(x=x.numpy().flatten(), y=y.numpy().flatten(), s=20, ax=ax_loc)

            ax_loc.set_title(title)
            ax_loc.set_xlabel(f'Observations')
            ax_loc.set_ylabel(f'Predictions')
            ax_loc.ticklabel_format(axis='both', style='sci', scilimits=(0, 4))
    
    if verbose:
        print (verbose_output)

    if save:
        plt.savefig(f'./figs/{save}.png', dpi=300, bbox_inches='tight')

def plot_gvi(data, axes, row, col, t_axis, label, color, size_samples):
    if data is not None:
        if len(data) > 10:
            axes[row][col].plot(
                t_axis, 
                data[:, row, col], 
                label=label, 
                color=color,
            )
        else:
            #* 0 nothing, 1 Circle, 2 triangle_up, 3 star, 
            #* 4 filled X, 5 tri_left, 6 square, 7 thin_diamond
            markers = ['none', 'o', '^', '*', 'X', '3', 's', 'd']

            for t_sample in range(len(size_samples)):
                axes[row][col].plot(
                    t_axis[t_sample], 
                    data[t_sample, row, col], 
                    label=label, 
                    color=color if size_samples[t_sample] != 0 else '#A9A9A9',
                    marker=markers[int(size_samples[t_sample]) if size_samples[t_sample] != 0 else 1],
                    linestyle='',
                    markersize=7
                )

def data_convert(target):
    return target[0].detach().cpu().numpy() if target[0] != None else None, \
        target[1].detach().cpu().numpy() if target[1] != None else None, \
        target[2].detach().cpu().numpy() if target[2] != None else None

def grid_visual_interpolate(
    model,
    raw_data: bool = True,
    save: bool = False
):
    fig, axes = plt.subplots(model.N.shape[1], model.N.shape[2], figsize=(40, 15), sharex=True)

    model = model.to('cpu')
    model.input_N = model.input_N.to('cpu')
    t_smoothed = torch.linspace(model.t_observed[0], model.t_observed[-1], 100).to('cpu')
    y_pred = model.eval_model(t_smoothed)

    #TODO fit for different data transformation techniques
    if raw_data:
        data_values = [model.N, torch.pow(y_pred, 4), None]
    else:
        data_values = [model.input_N, y_pred, None]

    obs, pred, pred2 = data_convert(data_values)
    t_obs, t_pred, t_pred2 = data_convert([model.t_observed, t_smoothed, None])
    data_names = ['Observations', 'Predictions', None]

    anno = pd.read_csv(os.path.join(model.data_dir, 'annotations.csv'))
    try: sample_N = model.sample_N.cpu().numpy() 
    except: sample_N = np.ones(model.N.shape)

    for row in range(model.N.shape[1]):
        for col in range(model.N.shape[2]):
            size_samples = sample_N[:, row, col]
            plot_gvi(pred2, axes, row, col, t_pred2, data_names[2], 'skyblue', size_samples)
            plot_gvi(pred, axes, row, col, t_pred, data_names[1], 'lightcoral', size_samples)
            plot_gvi(obs, axes, row, col, t_obs, data_names[0], '#2C6975', size_samples)
    
            axes[0][col].set_title(anno['populations'][col])
            axes[row][0].set_ylabel(anno['clones'][row])
            axes[row][col].set_xticks(t_obs, labels=t_obs.astype(int), rotation=45)
            axes[row][col].ticklabel_format(axis='y', style='sci', scilimits=(0, 4))

    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.5)

    #* 0 nothing, 1 Circle, 2 triangle_up, 3 star, 
    #* 4 filled X, 5 tri_left, 6 square, 7 thin_diamond
    markers = ['none', 'o', '^', '*', 'X', '3', 's', 'd']
    legend_elements = [Line2D([0], [0], color='lightcoral', lw=2)]
    labels = ['Predictions']
    for sample_id in range(1, int(np.max(sample_N)) + 1):
        legend_elements.append(
            Line2D(
                [0], [0], marker=markers[sample_id], color='#2C6975', 
                markersize=7, linestyle=''
            )
        )
        labels.append(f'Sampled {sample_id} time(s)' if int(np.max(sample_N)) > 1 else 'Observations')
    
    if int(np.min(sample_N)) == 0:
        legend_elements.insert(1, Line2D([0], [0], marker=markers[1], linestyle='', color='#A9A9A9', markersize=7))
        labels.insert(1, 'Sampled 0 time(s)')

    fig.legend(legend_elements, labels, loc='right', fontsize='x-large', bbox_to_anchor=(0.975, 0.5))

    if save:
        plt.savefig(f'./figs/{save}.png', dpi=300, bbox_inches='tight')

def transti_K(model, K, index=None, columns=None):
    anno = pd.read_csv(os.path.join(model.data_dir, 'annotations.csv'))
    transition_K = pd.DataFrame(
        index=anno['populations'].values if index is None else index, 
        columns=anno['populations'].values if columns is None else columns, 
        data=K
    )
    return transition_K

def clone_specific_K(model, K_type='const', index_clone=0, tpoint=1.0, sep='mixture', save=False):
    K = model.get_matrix_K(K_type=K_type, eval=True, tpoint=tpoint, sep=sep).detach().cpu().numpy()
    df = transti_K(model, K[index_clone])

    fig, axes = plt.subplots(figsize=(12, 6))
    sns.heatmap(df, annot=True, linewidths=.5, cmap='coolwarm', ax=axes, vmin=-2, vmax=3)
    plt.title(f'Transition Matrix for Clone {index_clone} Day {np.round(tpoint.cpu(), 1) if type(tpoint) == torch.Tensor else np.round(tpoint, 1)}')

    if save:
        plt.savefig(f'./figs/temp_{tpoint.round(decimals=1)}.png', dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
        plt.close()

def diff_K_between_clones(model, K_type='const', index_pop=0, tpoint=1.0, direction='outbound', sep='mixture'):
    K = model.get_matrix_K(K_type=K_type, eval=True, tpoint=tpoint, sep=sep).detach().cpu().numpy()
    anno = pd.read_csv(os.path.join(model.data_dir, 'annotations.csv'))
    pop_name = anno['populations'].values[index_pop]

    if direction == 'outbound':
        df = transti_K(model, K[:, index_pop, :], anno['clones'].values[:K.shape[0]], pop_name + ' -> ' + anno['populations'].values).T
    if direction == 'inbound':
        df = transti_K(model, K[:, :, index_pop], anno['clones'].values[:K.shape[0]], anno['populations'].values + ' -> ' + pop_name).T

    fig, axes = plt.subplots(figsize=(12, 6))
    sns.heatmap(df, annot=True, linewidths=.5, cmap='coolwarm', ax=axes, vmin=-2, vmax=3)
    plt.xticks(rotation=0)
    plt.title(f'Difference in transition rates between clones for {pop_name} ({direction})')

def rates_notin_paga(model, K_type='const', sep='mixture', value=False, save=False):
    K_total = []

    x = torch.linspace(0, int(model.t_observed[-1].item()), int(model.t_observed[-1].item() * 2 + 1)).to(model.config.gpu)
    for i in range(len(x)):
        K = model.get_matrix_K(K_type=K_type, eval=True, tpoint=x[i].round(decimals=1), sep=sep).detach().cpu().numpy()
        K = K[np.where(model.oppo_L.cpu().numpy())]
        K_total.append(K[np.where(K != 0)])

    sns.histplot((np.stack(K_total).flatten()), bins=50)
    plt.title(f'Rates not in PAGA that are non-zero')

    if save:
        plt.savefig(f'./figs/{save}.png', dpi=300, bbox_inches='tight')

    if value:
        return np.stack(K_total)

def rates_in_paga(model, K_type='const', sep='mixture', value=False, save=False):
    K_total = []

    x = torch.linspace(0, int(model.t_observed[-1].item()), int(model.t_observed[-1].item() * 2 + 1)).to(model.config.gpu)
    for i in range(len(x)):
        K = model.get_matrix_K(K_type=K_type, eval=True, tpoint=x[i].round(decimals=1), sep=sep).detach().cpu().numpy()
        K_total.append(K[np.where(np.broadcast_to(model.used_L.cpu().numpy(), K.shape))])
    
    sns.histplot(np.stack(K_total).flatten(), bins=50)
    plt.title(f'Rates in PAGA')

    if save:
        plt.savefig(f'./figs/{save}.png', dpi=300, bbox_inches='tight')

    if value:
        return np.stack(K_total)

def rates_diagonal(model, K_type='const', tpoint=1.0, sep='mixture'):
    K = model.get_matrix_K(K_type=K_type, eval=True, tpoint=tpoint, sep=sep)
    diag = torch.diagonal(K, dim1=-2, dim2=-1).detach().cpu().numpy()
    anno = pd.read_csv(os.path.join(model.data_dir, 'annotations.csv'))

    df = transti_K(model, diag, anno['clones'].values[:K.shape[0]]).T

    fig, axes = plt.subplots(figsize=(12, 6))
    sns.heatmap(df, annot=True, linewidths=.5, cmap='coolwarm', ax=axes, vmin=-2, vmax=3)
    plt.title(f'Diagonal of transition rates (Proliferation & Apoptosis)')

def clone_dynamic_K(model, K_type='const', index_clone=0, sep='mixture', suffix=''): 
    from PIL import Image
    x = torch.linspace(0, int(model.t_observed[-1].item()), int(model.t_observed[-1].item() * 2 + 1)).to(model.config.gpu)
    
    if index_clone == -1: title = 'BG'
    else: title = index_clone

    frames = []
    for i in range(len(x)):
        clone_specific_K(model, K_type, index_clone, x[i].round(decimals=1), sep, save=True)
        frames.append(Image.open(f'./figs/temp_{x[i].round(decimals=1)}.png'))
        os.remove(f'./figs/temp_{x[i].round(decimals=1)}.png')

    imageio.mimsave(f'K_dynamics_clone_{title}{suffix}.gif', frames, duration=500, loop=0)

def const_and_dyna(K_const, K_dyna, save=False):
    assert K_const.shape == K_dyna.shape
    from .pl import get_subplot_dimensions
    rows, cols, figsize = get_subplot_dimensions(K_const.shape[0], max_cols=3, fig_height_per_row=2)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for n in range(K_const.shape[0]):
        x = K_const[n].flatten()
        y = K_dyna[n].flatten()
        
        ax_loc = axes[n // cols][n % cols] if rows > 1 else axes[n]
        sns.scatterplot(x=x, y=y, s=25, ax=ax_loc, c='lightcoral')
        ax_loc.plot([x.min(), x.max()], [x.min(), x.max()], linestyle="--", color="grey")

        ax_loc.set_title(f'Clone {n}', fontsize=10)
        ax_loc.set_ylabel(f'K_dynamic')

        if n > 2:
            ax_loc.set_xlabel(f'K_const')

    if save is not False:
        plt.savefig(f'./figs/{save}.png', dpi=300, bbox_inches='tight')

def parameter_range(model_list, K_type='const', tpoint=1.0, sep='mixture', ref_model=None):
    pbar = tqdm(model_list)

    total_K = []
    for model in pbar:
        model.input_N = model.input_N.to('cpu')
        tpoint = torch.tensor([tpoint]).to('cpu')
        total_K.append(model.get_matrix_K(K_type=K_type, eval=True, tpoint=tpoint, sep=sep).detach().cpu().numpy())

    ref_K = ref_model.get_matrix_K(K_type=K_type, eval=True, tpoint=tpoint, sep=sep).detach().cpu().numpy()

    return np.stack(total_K), ref_K # (num_bootstraps, clones, pops, pops)

def parameter_ci(
    total_K, 
    data_dir: str = 'const', 
    index_clone: int = 0, 
    pop_1: int = 0,
    pop_2: int = 0, 
    ref_K: any = None,
    save: bool = False,
):
    anno = pd.read_csv(os.path.join(data_dir, 'annotations.csv'))
    
    sampled_ks = total_K[:, index_clone, pop_1, pop_2]
    lb, ub = np.percentile(sampled_ks, 2.5), np.percentile(sampled_ks, 97.5)

    g = sns.displot(sampled_ks, bins=int(len(sampled_ks) / 4), kde=True, color='#929591')
    g.fig.set_dpi(300)

    plt.axvline(ref_K[index_clone, pop_1, pop_2], linestyle='--', color='lightcoral')
    plt.axvline(lb, linestyle='--', color='#069AF3')
    plt.axvline(ub, linestyle='--', color='#069AF3')
    plt.title(f'Rate distributions of Clone {index_clone}', fontsize=10)
    plt.xlabel('From {} to {}'.format(anno['populations'].values[pop_1], anno['populations'].values[pop_2]), fontsize=10)

    #* (Original) (Bootstrapping {len(model_list)})
    legend_elements = [
        Line2D([0], [0], linestyle='--', color='lightcoral', lw=2), 
        Line2D([0], [0], linestyle='--', color='#069AF3', lw=2), 
    ]
    labels = ['Fitted', f'95% CI']
    plt.legend(legend_elements, labels, loc='best', fontsize=10)

    if save:
        plt.savefig(f'./figs/{save}.png', dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
        plt.close()

def trajectory_range(model_list, model_ref, raw_data=True):
    model_list.append(model_ref)
    pbar = tqdm(enumerate(model_list))
    total_pred = []

    for idx, model in pbar:
        model = model.to('cpu')
        model.input_N = model.input_N.to('cpu')
        t_smoothed = torch.linspace(model.t_observed[0], model.t_observed[-1], 100).to('cpu')
        y_pred = model.eval_model(t_smoothed)

        #TODO fit for different data transformation techniques
        if raw_data:
            data_values = [model.N, torch.pow(y_pred, 4), None]
        else:
            data_values = [model.input_N, y_pred, None]

        obs, pred, _ = data_convert(data_values)
        total_pred.append(pred)
    
    return np.stack(total_pred), obs, t_smoothed

def trajectory_ci(
    total_pred,
    obs,
    t_smoothed,
    model_ref,
    boundary,
    save: bool = False
):
    fig, axes = plt.subplots(model_ref.N.shape[1], model_ref.N.shape[2], figsize=(40, 15), sharex=True)
    lb, ub = np.percentile(total_pred, boundary[0], axis=0), np.percentile(total_pred, boundary[1], axis=0)

    t_obs, t_pred, t_median = data_convert([model_ref.t_observed, t_smoothed, t_smoothed])
    data_names = ['Observations', 'Predictions', 'Q50']
    anno = pd.read_csv(os.path.join(model_ref.data_dir, 'annotations.csv'))

    for row in range(model_ref.N.shape[1]):
        for col in range(model_ref.N.shape[2]):
            axes[row][col].fill_between(
                t_pred,
                lb[:, row, col],
                ub[:, row, col],
                label='1 std error',
                color='lightskyblue',
                alpha=0.5
            )

            plot_gvi(np.percentile(total_pred, 50, axis=0), axes, row, col, t_median, data_names[2], '#929591', np.array([1., 1., 1., 1.]))
            plot_gvi(total_pred[-1], axes, row, col, t_pred, data_names[1], 'lightcoral', np.array([1., 1., 1., 1.]))
            plot_gvi(obs, axes, row, col, t_obs, data_names[0], '#2C6975', np.array([1., 1., 1., 1.]))
    
            axes[0][col].set_title(anno['populations'][col])
            axes[row][0].set_ylabel(anno['clones'][row])
            axes[row][col].set_xticks(t_obs, labels=t_obs.astype(int), rotation=45)
            axes[row][col].ticklabel_format(axis='y', style='sci', scilimits=(0, 4))

    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.5)

    legend_elements = [
        Line2D([0], [0], marker='o', color='#2C6975', markersize=7, linestyle=''), 
        Line2D([0], [0], color='lightcoral', lw=2), 
        Line2D([0], [0], color='lightskyblue', lw=2), 
        Line2D([0], [0], color='#929591', lw=2)
    ]
    labels = ['Observations', 'Predictions', f'Q{boundary[0]} - Q{boundary[1]}', 'Q50']
    fig.legend(legend_elements, labels, loc='right', fontsize='x-large', bbox_to_anchor=(0.96, 0.5))

    if save:
        plt.savefig(f'./figs/{save}.png', dpi=300, bbox_inches='tight')