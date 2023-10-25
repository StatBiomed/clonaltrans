from scipy.stats import spearmanr, pearsonr
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
import scipy.stats as stats
from scipy.interpolate import interp1d

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
        fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] + 5, figsize[1]))

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

def get_variance(model):
    std_reshape = torch.broadcast_to(model.ode_func.std, model.input_N.shape).clone().detach().cpu().numpy()
    t_smoothed = np.linspace(model.t_observed[0].detach().cpu().numpy(), model.t_observed[-1].detach().cpu().numpy(), 100)

    std_reshape[1] = std_reshape[1] * 5.354
    std_reshape[2] = std_reshape[2] * 583.204
    std_reshape[3] = std_reshape[3] * 635.470

    std_inferred = np.zeros((100, model.N.shape[1], model.N.shape[2]))

    from itertools import product
    for c, p in product(range(model.N.shape[1]), range(model.N.shape[2])):
        x = std_reshape[:, c, p]
        f = interp1d([0, 3, 10, 17], x, kind='quadratic')
        y_smoothed = f(t_smoothed)
        std_inferred[:, c, p] = y_smoothed

    return std_inferred

def grid_visual_interpolate(
    model,
    raw_data: bool = True,
    variance: bool = False,
    device: str = 'cpu',
    save: bool = False
):
    fig, axes = plt.subplots(model.N.shape[1], model.N.shape[2], figsize=(45, 20), sharex=True)

    model = model.to(device)
    model.input_N = model.input_N.to(device)
    t_smoothed = torch.linspace(model.t_observed[0], model.t_observed[-1], 100).to(device)
    y_pred = model.eval_model(t_smoothed)

    #TODO fit for different data transformation techniques
    if raw_data:
        data_values = [model.N, torch.pow(y_pred, 1 / model.config.exponent), None]
    else:
        data_values = [model.input_N, y_pred, None]
    
    if variance:
        std_inferred = get_variance(model)
        lb, ub = y_pred.cpu().numpy() - std_inferred, y_pred.cpu().numpy() + std_inferred
        lb, ub = np.clip(lb, 0, np.max(lb)), np.clip(ub, 0, np.max(ub))

        if raw_data:
            lb, ub = np.power(lb, 1 / model.config.exponent), np.power(ub, 1 / model.config.exponent)

    obs, pred, pred2 = data_convert(data_values)
    t_obs, t_pred, t_pred2 = data_convert([model.t_observed, t_smoothed, None])
    data_names = ['Observations', 'Predictions', None]

    anno = pd.read_csv(os.path.join(model.data_dir, 'annotations.csv'))
    try: sample_N = model.sample_N.cpu().numpy() 
    except: sample_N = np.ones(model.N.shape)

    for row in range(model.N.shape[1]):
        for col in range(model.N.shape[2]):
            if variance:
                axes[row][col].fill_between(
                    t_pred,
                    lb[:, row, col],
                    ub[:, row, col],
                    color='lightskyblue',
                    alpha=0.5
                )

            size_samples = sample_N[:, row, col]
            plot_gvi(pred2, axes, row, col, t_pred2, data_names[2], 'skyblue', size_samples)
            plot_gvi(pred, axes, row, col, t_pred, data_names[1], 'lightcoral', size_samples)
            plot_gvi(obs, axes, row, col, t_obs, data_names[0], '#2C6975', size_samples)
    
            axes[0][col].set_title(anno['populations'][col], fontsize=15)
            axes[row][0].set_ylabel(anno['clones'][row], fontsize=15)
            axes[row][col].set_xticks(t_obs, labels=t_obs.astype(int), rotation=45)
            # axes[row][col].set_yticks([])
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

    if variance:
        legend_elements.append(Line2D([0], [0], color='lightskyblue', lw=4))
        labels.append('mean $\pm$ 1 std')

    fig.legend(legend_elements, labels, loc='right', fontsize=15, bbox_to_anchor=(0.97, 0.5))

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

def transit_K(model, K, index=None, columns=None):
    anno = pd.read_csv(os.path.join(model.data_dir, 'annotations.csv'))
    transition_K = pd.DataFrame(
        index=anno['populations'].values[:model.N.shape[2]] if index is None else index, 
        columns=anno['populations'].values[:model.N.shape[2]] if columns is None else columns, 
        data=K
    )
    return transition_K

def compare_with_bg(model, K_type='const', save=False):
    K_total = []

    x = torch.linspace(0, int(model.t_observed[-1].item()), int(model.t_observed[-1].item() + 1)).to(model.config.gpu)
    for i in range(len(x)):
        K = model.get_matrix_K(K_type=K_type, eval=True, tpoint=x[i].round()).detach().cpu().numpy()
        K_total.append(K)
    
    K_total = np.stack(K_total)
    x = np.mean(K_total[:, :-1, :, :], axis=1).flatten()
    y = K_total[:, -1, :, :].flatten()

    corr, p_value = pearsonr(x, y)
    ax = sns.scatterplot(x=y, y=x, s=20, c='#2c6aab')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot([x.min(), x.max()], [x.min(), x.max()], linestyle="--", color="grey")
    plt.xlabel(f'All cells (incl. filtered at pre-processing)', fontsize=13)
    plt.ylabel(f'Mean of {K_total.shape[1] - 1} meta-clones', fontsize=13)
    plt.text(1.6, -0.3, f'$Pearson \; r = {corr:.3f}$', fontsize=13)
    plt.yticks([-0.5, 0, 1, 2], [-0.5, 0, 1, 2], fontsize=13)
    plt.xticks([-0.5, 0, 1, 2, 3], [-0.5, 0, 1, 2, 3], fontsize=13)
    plt.title(f'Comparison of rates (Day 0 ~ Day 17)', fontsize=13)

    if save is not False:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

def clone_specific_K(model, K_type='const', index_clone=0, tpoint=1.0, save=False):
    K = model.get_matrix_K(K_type=K_type, eval=True, tpoint=tpoint).detach().cpu().numpy()
    df = transit_K(model, K[index_clone])

    if tpoint in model.t_observed:
        N = model.N[torch.where(model.t_observed == tpoint)[0][0], index_clone]
    else:
        N = model.eval_model(torch.Tensor([0.0, tpoint]))[1, index_clone]
        N = torch.pow(N, 1 / model.config.exponent)
    
    # df[np.abs(df) < 1e-4] = 0
    # df.iloc[list(np.where(N < 0.5)[0])] = 0

    fig, axes = plt.subplots(1, 1, figsize=(16, 6))
    ax = sns.heatmap(
        df, annot=True, linewidths=.5, cmap='coolwarm', 
        ax=axes, vmin=-2, vmax=3
    )
    
    title = f'Transition rates for Clone {index_clone} | Day {np.round(tpoint, 1)}' \
        if K_type == 'dynamic' else f'Transition rates for clone {index_clone}'
    plt.title(title, fontsize=13)

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

def diff_K_between_clones(model, K_type='const', index_pop=0, tpoint=1.0, direction='outbound'):
    K = model.get_matrix_K(K_type=K_type, eval=True, tpoint=tpoint).detach().cpu().numpy()
    anno = pd.read_csv(os.path.join(model.data_dir, 'annotations.csv'))
    pop_name = anno['populations'].values[index_pop]

    if direction == 'outbound':
        df = transit_K(model, K[:, index_pop, :], anno['clones'].values[:K.shape[0]], pop_name + ' -> ' + anno['populations'].values).T
    if direction == 'inbound':
        df = transit_K(model, K[:, :, index_pop], anno['clones'].values[:K.shape[0]], anno['populations'].values + ' -> ' + pop_name).T

    fig, axes = plt.subplots(figsize=(12, 6))
    sns.heatmap(df, annot=True, linewidths=.5, cmap='coolwarm', ax=axes, vmin=-2, vmax=3)
    plt.xticks(rotation=0)
    plt.title(f'Difference in transition rates between clones for {pop_name} ({direction})')

def rates_notin_paga(model, K_type='const', value=False, save=False):
    K_total = []

    x = torch.linspace(0, int(model.t_observed[-1].item()), int(model.t_observed[-1].item() + 1)).to(model.config.gpu)
    for i in range(len(x)):
        K = model.get_matrix_K(K_type=K_type, eval=True, tpoint=x[i].round(decimals=1)).detach().cpu().numpy()
        K = K[np.where(model.oppo_L.cpu().numpy())]
        K_total.append(K[np.where(K != 0)])

    sns.histplot(np.stack(K_total).flatten() if K_type != 'const' else K_total[0].flatten(), bins=50)
    plt.title(f'Rates not in PAGA that are non-zero')

    if save:
        plt.savefig(f'./figs/{save}.png', dpi=300, bbox_inches='tight', transparent=False, facecolor='white')

    if value:
        return np.stack(K_total)

def rates_in_paga(model, K_type='const', value=False, save=False):
    K_total = []

    x = torch.linspace(0, int(model.t_observed[-1].item()), int(model.t_observed[-1].item() + 1)).to(model.config.gpu)
    # x = torch.linspace(0, 17, 18).to(model.config.gpu)
    for i in range(len(x)):
        K = model.get_matrix_K(K_type=K_type, eval=True, tpoint=x[i].round()).detach().cpu().numpy()
        K_total.append(K[np.where(np.broadcast_to(model.used_L.cpu().numpy(), K.shape))])
    
    if K_type == 'const':
        K_total = K_total[0]

    ax = sns.histplot(np.stack(K_total).flatten(), bins=50)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize=13)
    # plt.yticks([0, 800, 1600, 2400], [0, 800, 1600, 2400], fontsize=13)
    plt.ylabel('Count', fontsize=13)
    plt.xlabel(f'Per capita transition rates of {model.N.shape[1] - 1} meta-clones', fontsize=13)
    plt.title(f'Distribution of rates (Day 0 ~ Day 17)', fontsize=13)
    plt.text(1.4, 1800, f'Each meta-clone has:\n16 proliferation & \n20 differentiation rates', fontsize=13)

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

    if value:
        return np.stack(K_total)

def rates_diagonal(model, K_type='const', tpoint=1.0):
    K = model.get_matrix_K(K_type=K_type, eval=True, tpoint=tpoint)
    diag = torch.diagonal(K, dim1=-2, dim2=-1).detach().cpu().numpy()
    anno = pd.read_csv(os.path.join(model.data_dir, 'annotations.csv'))

    df = transit_K(model, diag, anno['clones'].values[:K.shape[0]]).T

    fig, axes = plt.subplots(figsize=(12, 6))
    sns.heatmap(df, annot=True, linewidths=.5, cmap='coolwarm', ax=axes, vmin=-2, vmax=3)
    plt.title(f'Diagonal of transition rates (Proliferation & Apoptosis)')

def clone_dynamic_K(model, K_type='const', index_clone=0, suffix=''): 
    from PIL import Image
    x = torch.linspace(0, int(model.t_observed[-1].item()), int(model.t_observed[-1].item() + 1)).to(model.config.gpu)
    
    if index_clone == -1: title = 'BG'
    else: title = index_clone

    frames = []
    for i in range(len(x)):
        clone_specific_K(model, K_type, index_clone, x[i].round(decimals=1), save=True)
        frames.append(Image.open(f'./figs/temp_{x[i].round(decimals=1)}.png'))
        os.remove(f'./figs/temp_{x[i].round(decimals=1)}.png')

    imageio.mimsave(f'K_dynamics_clone_{title}{suffix}.gif', frames, duration=500, loop=0)

def clone_dynamic_K_lines(model, index_clone=0, save=False):
    K_total = []
    N_total = []
    t_obs = model.t_observed.cpu().numpy()
    x = torch.linspace(0, int(model.t_observed[-1].item()), int(model.t_observed[-1].item()) + 1).to(0)

    model.t_observed = model.t_observed.to(0)
    
    for i in range(len(x)):
        K = model.get_matrix_K(K_type=model.config.K_type, eval=True, tpoint=x[i]).detach().cpu().numpy()
        K_total.append(K)

        N = model.eval_model(torch.Tensor([0.0, x[i] + 0.01 if x[i] == 0 else x[i]]))[1, index_clone]
        N = torch.pow(N, 1 / model.config.exponent)
        
        N_total.append(N)

    K_total = np.stack(K_total)
    N_total = np.stack(N_total)

    anno = pd.read_csv(os.path.join(model.data_dir, 'annotations.csv'))
    graph = pd.read_csv(os.path.join(model.data_dir, 'graph_table.csv'), index_col=0)
    np.fill_diagonal(graph.values, 1)

    rows, cols, figsize = get_subplot_dimensions(np.sum(graph.values != 0), fig_height_per_row=4)
    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] + 5, figsize[1]))

    count = 0
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if graph.values[i][j] != 0:   
                df = K_total[:, index_clone, i, j]             
                df[np.abs(df) < 1e-4] = 0
                df[np.where(N_total[:, i] < 0.5)[0]] = 0

                f = interp1d(np.linspace(0, int(model.t_observed[-1].item()), int(model.t_observed[-1].item()) + 1), df)
                newx = np.linspace(0, int(model.t_observed[-1].item()), 50)
                newy = f(newx)

                axes[count // cols][count % cols].plot(
                    newx, 
                    newy, 
                    color='#2C6975',
                    lw=4,
                )

                axes[count // cols][0].set_ylabel('Per capita transition rate', fontsize=13)
                axes[count // cols][count % cols].set_title('From {} to {}'.format(anno['populations'].values[i], anno['populations'].values[j]), fontsize=13)
                axes[count // cols][count % cols].set_xticks(t_obs, labels=t_obs.astype(int))
                count += 1
    
    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

def get_dynamic_K_lines_conf(model_list, model=None, K_type='const'):
    K_total = []
    ref_K_total = []
    N_total = []

    t_obs = model.t_observed.cpu().numpy()
    x = torch.linspace(0, int(model.t_observed[-1].item()), int(model.t_observed[-1].item()) + 1).to('cpu')
    
    pbar = tqdm(range(len(x)))
    for i in pbar:
        total_K, ref_K = parameter_range(model_list, K_type=K_type, tpoint=x[i], ref_model=model)

        K_total.append(total_K) 
        ref_K_total.append(ref_K) 

        if x[i] in model.t_observed:
            N = model.N[torch.where(model.t_observed == x[i])[0][0]]
        else:
            N = model.eval_model(torch.Tensor([0.0, x[i] + 0.01 if x[i] == 0 else x[i]]))[1]
            N = torch.pow(N, 1 / model.config.exponent)
        
        N_total.append(N)
    
    K_total = np.stack(K_total) #* (time, num_bootstraps, clones, pops, pops)
    N_total = np.stack(N_total) #* (time, clones, pops)
    ref_K_total = np.stack(ref_K_total) #* (time, clones, pops, pops)
    print (K_total.shape, N_total.shape, ref_K_total.shape)
    return K_total, ref_K_total, N_total

def interpolate_1d(model, y):
    f = interp1d(np.linspace(0, int(model.t_observed[-1].item()), int(model.t_observed[-1].item()) + 1), y)
    newx = np.linspace(0, int(model.t_observed[-1].item()), 50)
    newy = f(newx)
    return newx, newy

def clone_dynamic_K_lines_conf(K_total, ref_K_total, N_total, model=None, index_clone=0, save=False):
    from itertools import product
    t_obs = model.t_observed.cpu().numpy()
    anno = pd.read_csv(os.path.join(model.data_dir, 'annotations.csv'))
    graph = pd.read_csv(os.path.join(model.data_dir, 'graph_table.csv'), index_col=0)
    np.fill_diagonal(graph.values, 1)

    rows, cols, figsize = get_subplot_dimensions(np.sum(graph.values != 0), fig_height_per_row=4)
    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] + 5, figsize[1]))

    K_total[np.abs(K_total) < 1e-4] = 0
    ref_K_total[np.abs(ref_K_total) < 1e-4] = 0

    line_lb, line_ub, line_mean = [], [], []

    count = 0
    for i, j in product(range(graph.shape[0]), range(graph.shape[1])):
        if graph.values[i][j] != 0:
            for idx_time in range(K_total.shape[0]):
                sub_K_total = K_total[idx_time, :, index_clone, i, j]
                lb, ub = np.percentile(sub_K_total, 2.5), np.percentile(sub_K_total, 97.5)
                mean = ref_K_total[idx_time, index_clone, i, j]

                if N_total[idx_time, index_clone, i] < 0.5:
                    lb, ub, mean = 0, 0, 0

                line_lb.append(lb)
                line_ub.append(ub)
                line_mean.append(mean)

            _, lby = interpolate_1d(model, line_lb)
            _, uby = interpolate_1d(model, line_ub)
            meanx, meany = interpolate_1d(model, line_mean)

            axes[count // cols][count % cols].fill_between(
                meanx,
                lby,
                uby,
                color='lightskyblue',
                alpha=0.5
            )

            axes[count // cols][count % cols].plot(
                meanx, 
                meany, 
                color='#2C6975',
                lw=4,
            )
            
            axes[count // cols][0].set_ylabel('Per capita transition rate', fontsize=13)
            axes[count // cols][count % cols].set_title('From {} to {}'.format(anno['populations'].values[i], anno['populations'].values[j]), fontsize=13)
            axes[count // cols][count % cols].set_xticks(t_obs, labels=t_obs.astype(int))
            
            count += 1
            line_lb, line_ub, line_mean = [], [], []
    
    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

def parameter_range(model_list, K_type='const', tpoint=1.0, ref_model=None):
    total_K = []
    for model in model_list:
        model.input_N = model.input_N.to('cpu')
        model.ode_func.supplement = [model.ode_func.supplement[i].to('cpu') for i in range(4)]
        tpoint = torch.tensor([tpoint]).to('cpu')
        total_K.append(model.get_matrix_K(K_type=K_type, eval=True, tpoint=tpoint).detach().cpu().numpy())

    ref_K = ref_model.get_matrix_K(K_type=K_type, eval=True, tpoint=tpoint).detach().cpu().numpy()

    return np.stack(total_K), ref_K # (num_bootstraps, clones, pops, pops)

def parameter_ci(
    model_list, 
    ref_model, 
    K_type: str = 'const',
    index_clone: int = 0, 
    pop_1: int = 0,
    pop_2: int = 0, 
    tpoint: float = 3.0,
    save: bool = False,
):
    if tpoint in ref_model.t_observed:
        N = ref_model.N[torch.where(ref_model.t_observed == tpoint)[0][0]]
    else:
        N = ref_model.eval_model(torch.Tensor([0.0, tpoint + 0.01 if tpoint == 0 else tpoint]))[1]
        N = torch.pow(N, 1 / ref_model.config.exponent)

    total_K, ref_K = parameter_range(model_list, K_type, tpoint, ref_model=ref_model)
    anno = pd.read_csv(os.path.join(ref_model.data_dir, 'annotations.csv'))

    sampled_ks = total_K[:, index_clone, pop_1, pop_2]
    print (N[index_clone, pop_1])
    if N[index_clone, pop_1] < 0.5:
        sampled_ks = np.zeros(sampled_ks.shape)
        
    lb, ub = np.percentile(sampled_ks, 2.5), np.percentile(sampled_ks, 97.5)
    print (sampled_ks)

    g = sns.displot(sampled_ks, kde=True, color='#929591')
    g.fig.set_dpi(600)

    if N[index_clone, pop_1] >= 0.5:
        plt.axvline(ref_K[index_clone, pop_1, pop_2], linestyle='--', color='lightcoral')
    else:
        plt.axvline(0, linestyle='--', color='lightcoral')
    
    plt.axvline(lb, linestyle='--', color='#069AF3')
    plt.axvline(ub, linestyle='--', color='#069AF3')

    title = f'Bootstrapping rates for Clone {index_clone} | Day {np.round(tpoint, 1)}' \
        if K_type == 'dynamic' else f'Transition rates for clone {index_clone}'
    plt.title(title, fontsize=13)

    plt.xlabel('From {} to {}'.format(anno['populations'].values[pop_1], anno['populations'].values[pop_2]), fontsize=12)
    plt.xticks(fontsize=14)
    plt.ylabel('# of trails', fontsize=14)
    plt.yticks([])

    #* (Original) (Bootstrapping {len(model_list)})
    legend_elements = [
        Line2D([0], [0], linestyle='--', color='lightcoral', lw=2), 
        Line2D([0], [0], linestyle='--', color='#069AF3', lw=2), 
    ]
    labels = ['Fitted', f'95% CI']
    plt.legend(legend_elements, labels, loc='best', fontsize=12)

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')
        plt.close()

def test_diffclones(total_K, ref_model):
    paga = pd.read_csv(os.path.join(ref_model.data_dir, 'graph_table.csv'), index_col=0).astype(np.int32).values
    np.fill_diagonal(paga, 1)

    shapiro_res = np.zeros((28, ref_model.N.shape[2], ref_model.N.shape[2]))
    overall_res = np.zeros((28, ref_model.N.shape[2], ref_model.N.shape[2]))
    shapiro_res[shapiro_res == 0] = 'nan'
    overall_res[overall_res == 0] = 'nan'

    count = 0
    for idx_c1 in range(ref_model.N.shape[1] - 2):
        for idx_c2 in range(idx_c1 + 1, ref_model.N.shape[1] - 1):
            # print (f'Test clone {idx_c1, idx_c2}.')

            for pop1 in range(ref_model.N.shape[2]):
                for pop2 in range(ref_model.N.shape[2]):

                    if paga[pop1, pop2] == 1:
                        K_c1 = total_K[:, idx_c1, pop1, pop2]
                        K_c2 = total_K[:, idx_c2, pop1, pop2]

                        K_c1 = K_c1[(K_c1 > np.percentile(K_c1, 2.5)) & (K_c1 < np.percentile(K_c1, 97.5))]
                        K_c2 = K_c2[(K_c2 > np.percentile(K_c2, 2.5)) & (K_c2 < np.percentile(K_c2, 97.5))]

                        _, shapiro_p_c1 = stats.shapiro(K_c1)
                        _, shapiro_p_c2 = stats.shapiro(K_c2)

                        if shapiro_p_c1 > 0.05 and shapiro_p_c2 > 0.05:
                        # if pop1 == pop2:
                            _, paired_p = stats.ttest_ind(K_c1, K_c2) 
                            overall_res[count, pop1, pop2] = paired_p
                        else:
                            _, wilcox_p = stats.mannwhitneyu(K_c1, K_c2)
                            overall_res[count, pop1, pop2] = wilcox_p
            count += 1
    
    return shapiro_res, overall_res

def plot_diffclones(
    overall_res,
    ref_model,
    save=False
):
    anno = pd.read_csv(os.path.join(ref_model.data_dir, 'annotations.csv'))
    graph = pd.read_csv(os.path.join(ref_model.data_dir, 'graph_table.csv'), index_col=0)
    np.fill_diagonal(graph.values, 1)

    # temp = overall_res.reshape((overall_res.shape[0], overall_res.shape[1] * overall_res.shape[1]))
    # nan_cols = np.isnan(temp).any(axis=0)
    # temp = temp[:, ~nan_cols]
    
    cols = []
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if graph.values[i][j] != 0:
                cols.append('{} -> {}'.format(anno['populations'].values[i], anno['populations'].values[j]))

    fig, axes = plt.subplots(figsize=(22, 10))
    overall_res[overall_res > 0.05] = np.nan

    index = []
    for idx_c1 in range(ref_model.N.shape[1] - 2):
        for idx_c2 in range(idx_c1 + 1, ref_model.N.shape[1] - 1):
            index.append(f'Clone {idx_c1} / {idx_c2}')
    df = pd.DataFrame(data=-np.log10(overall_res.T), index=cols, columns=index)
    ax = sns.heatmap(df, annot=False, linewidths=.5, cmap='viridis', vmin=0, vmax=60)
    plt.title('$-log_{10}$ p-value across meta-clones & populations', fontsize=18)
    plt.xticks(rotation=30, fontsize=14)
    plt.yticks(fontsize=14)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')