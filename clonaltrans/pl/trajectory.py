import torch
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
import os
from itertools import product
from utils import get_boots_K_total
import seaborn as sns
from .base import get_subplot_dimensions, interpolate_1d

def plot_grid(data, axes, row, col, t_axis, label, color, size_samples, markers):
    if len(data) > 10:
        axes[row][col].plot(
            t_axis, 
            data[:, row, col], 
            label=label, 
            color=color,
        )

    else:
        for idx, t_sample in enumerate(size_samples):
            axes[row][col].plot(
                t_axis[idx], 
                data[idx, row, col], 
                label=label, 
                color=color if t_sample != 0 else '#A9A9A9',
                marker=markers[int(t_sample) if t_sample != 0 else 1],
                linestyle='',
                markersize=7
            )

def get_variance(model, t_predictions):
    std_reshape = torch.broadcast_to(model.model.block.std, model.N.shape).clone().detach().cpu().numpy()
    std_inferred = np.zeros((100, model.N.shape[1], model.N.shape[2]))

    for idx, fac in enumerate(model.scaling_factor):
        std_reshape[idx] *= fac

    for c, p in product(range(model.N.shape[1]), range(model.N.shape[2])):
        x = std_reshape[:, c, p]
        f = interp1d(model.t_observed.detach().cpu().numpy(), x, kind='quadratic')
        std_inferred[:, c, p] = f(t_predictions)

    return std_inferred

def grid_visualize(
    model,
    device: str = 'cpu',
    save: bool = False
):
    model = model.to(device)
    fig, axes = plt.subplots(model.N.shape[1], model.N.shape[2], figsize=(45, 20), sharex=True)

    t_smoothed = torch.linspace(model.t_observed[0], model.t_observed[-1], 100).to(device)
    y_pred = model.eval_model(t_smoothed)

    observations = model.N.detach().cpu().numpy()
    predictions = y_pred.detach().cpu().numpy()
    t_observations = model.t_observed.detach().cpu().numpy()
    t_predictions = t_smoothed.detach().cpu().numpy()

    std_inferred = get_variance(model, t_predictions)
    lb, ub = predictions - std_inferred, predictions + std_inferred
    lb, ub = np.clip(lb, 0, np.max(lb)), np.clip(ub, 0, np.max(ub))

    anno = pd.read_csv(os.path.join(model.config['data_loader']['args']['data_dir'], model.config['data_loader']['args']['annots']))
    try: sample_N = model.sample_N.cpu().numpy() 
    except: sample_N = np.ones(model.N.shape)

    #* 0 nothing, 1 Circle, 2 triangle_up, 3 star, 
    #* 4 filled X, 5 tri_left, 6 square, 7 thin_diamond
    markers = ['none', 'o', '^', '*', 'X', '3', 's', 'd']

    for row, col in product(range(model.N.shape[1]), range(model.N.shape[2])):
        axes[row][col].fill_between(
            t_predictions,
            lb[:, row, col],
            ub[:, row, col],
            color='lightskyblue',
            alpha=0.5
        )

        size_samples = sample_N[:, row, col]
        plot_grid(predictions, axes, row, col, t_predictions, 'Predictions', 'lightcoral', size_samples, markers)
        plot_grid(observations, axes, row, col, t_observations, 'Observations', '#2C6975', size_samples, markers)

        axes[0][col].set_title(anno['populations'][col], fontsize=15)
        axes[row][0].set_ylabel(anno['clones'][row], fontsize=15)
        axes[row][col].set_xticks(t_observations, labels=t_observations.astype(int), rotation=45)
        axes[row][col].ticklabel_format(axis='y', style='sci', scilimits=(0, 4))

    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.5)

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

    legend_elements.append(Line2D([0], [0], color='lightskyblue', lw=4))
    labels.append('mean $\pm$ 1 std')

    fig.legend(legend_elements, labels, loc='right', fontsize=15, bbox_to_anchor=(0.97, 0.5))

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

def parameter_ci(
    model_list, 
    ref_model, 
    index_clone: int = 0, 
    pop_1: int = 0,
    pop_2: int = 0, 
    tpoint: float = 3.0,
    save: bool = False,
):
    tpoint = torch.tensor([tpoint]).to('cpu')
    K_type = ref_model.config['arch']['args']['K_type']
    anno = pd.read_csv(os.path.join(ref_model.config['data_loader']['args']['data_dir'], ref_model.config['data_loader']['args']['annots']))

    total_K, ref_K = get_boots_K_total(model_list, ref_model, K_type, tpoint)
    N = ref_model.eval_model(torch.tensor([0.0, max(tpoint, 0.01)]))[1]

    bootstrap_K = total_K[:, index_clone, pop_1, pop_2] if N[index_clone, pop_1] > 0.5 else np.zeros(total_K.shape[0])
    lb, ub = np.percentile(bootstrap_K, 2.5), np.percentile(bootstrap_K, 97.5)

    g = sns.displot(bootstrap_K, kde=True, color='#929591')
    # g.fig.set_dpi(300)

    if N[index_clone, pop_1] >= 0.5: plt.axvline(ref_K[index_clone, pop_1, pop_2], linestyle='--', color='lightcoral')
    else: plt.axvline(0, linestyle='--', color='lightcoral')

    title = f'Bootstrapping rates for Clone {index_clone} | Day {np.round(tpoint.item(), 1)}' \
        if K_type == 'dynamic' else f'Bootstrapping rates for clone {index_clone}'
    plt.title(title, fontsize=12)

    plt.axvline(lb, linestyle='--', color='#069AF3')
    plt.axvline(ub, linestyle='--', color='#069AF3')
    plt.xlabel('From {} to {}'.format(anno['populations'].values[pop_1], anno['populations'].values[pop_2]), fontsize=12)
    plt.xticks(fontsize=12)
    plt.ylabel('# of trails', fontsize=12)
    plt.yticks([])

    legend_elements = [
        Line2D([0], [0], linestyle='--', color='lightcoral', lw=2), 
        Line2D([0], [0], linestyle='--', color='#069AF3', lw=2), 
    ]
    labels = ['Fitted', f'95% CI']
    plt.legend(legend_elements, labels, loc='best', fontsize=12)

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

