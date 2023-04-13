from scipy.stats import spearmanr
from torch import nn
import matplotlib.pyplot as plt
import math
import seaborn as sns
import numpy as np
import pandas as pd

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

def eval_predictions(model, t_observed, save=False):
    from .pl import mse_corr
    t_observed_norm = (t_observed - t_observed[0]) / (t_observed[-1] - t_observed[0])

    observations = model.N
    predictions = model.eval_model(t_observed_norm)

    mse_corr(observations[1:], predictions[1:], t_observed[1:].cpu().numpy(), save=save)

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

    for n in range(num_t):
        loss = SmoothL1(observations[n], predictions[n])
        x = observations[n].cpu().numpy().flatten()
        y = predictions[n].detach().cpu().numpy().flatten()
        spear = spearmanr(x, y)[0]
        
        ax_loc = axes[n % cols][n // cols] if rows > 1 else axes[n]
        sns.scatterplot(
            x=x, y=y, s=size, ax=ax_loc,
            hue=None, palette=None
        )
        ax_loc.set_title(f'Time {t_observed[n]} Loss {loss.item():.3f} Corr {spear:.3f}')
        ax_loc.set_xlabel(f'Observations')
        ax_loc.set_ylabel(f'Predictions')

    if save:
        plt.savefig(f'./figs/{save}.png', dpi=300, bbox_inches='tight')

def grid_visual_interpolate(
    observations, 
    predictions, 
    t_observed, 
    t_pred=None, 
    save=False
):
    if t_pred == None:
        t_pred = t_observed.clone()
        
    fig, axes = plt.subplots(observations.shape[1], observations.shape[2], figsize=(33, 20), sharex=True)
    obs = observations.cpu().numpy()
    pred = predictions.detach().cpu().numpy()

    anno = pd.read_csv('./data/annotations.csv')

    for row in range(observations.shape[1]):
        for col in range(observations.shape[2]):
            axes[row][col].plot(
                t_pred.cpu().numpy(), 
                pred[:, row, col], 
                label='Predictions', 
                color='#CDB3D4',
            )

            axes[row][col].plot(
                t_observed.cpu().numpy(), 
                obs[:, row, col], 
                label='Observations', 
                color='#2C6975', 
                marker='o',
                linestyle='',
                markersize=5
            )
            axes[row][col].set_xticks(t_observed.cpu().numpy(), labels=t_observed.cpu().numpy().astype(int), rotation=45)
    
            axes[0][col].set_title(anno['populations'][col])
            axes[row][0].set_ylabel(anno['clones'][row])

    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.5)
    handles, labels = axes[row][col].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), fontsize='x-large')

    if save:
        plt.savefig(f'./figs/{save}.png', dpi=300, bbox_inches='tight')