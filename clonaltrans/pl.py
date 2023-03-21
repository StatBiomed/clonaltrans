from scipy.stats import spearmanr
from torch import nn
import matplotlib.pyplot as plt
import math
import seaborn as sns
from scipy.interpolate import interp1d
import numpy as np

MSE = nn.MSELoss(reduction='mean')

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

def eval_predictions(model, t_observed):
    from .pl import mse_corr, grid_visual

    observations = model.N
    predictions = model.eval_model(t_observed)

    mse_corr(observations[1:], predictions[1:], t_observed[1:].cpu().numpy())
    # grid_visual(observations, predictions, t_observed)

def mse_corr(observations, predictions, t_observed, size=20, hue=None, palette=None):
    num_t = observations.shape[0]
    print (f'There are {num_t} observed timepoints except the inital time.')

    from .pl import get_subplot_dimensions
    rows, cols, figsize = get_subplot_dimensions(num_t, fig_height_per_row=4)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for n in range(num_t):
        loss = MSE(observations[n], predictions[n])
        x = observations[n].cpu().numpy().flatten()
        y = predictions[n].detach().cpu().numpy().flatten()
        spear = spearmanr(x, y)[0]
        
        ax_loc = axes[n % cols][n // cols] if rows > 1 else axes[n]
        sns.scatterplot(
            x=x, y=y, s=size, ax=ax_loc,
            hue=None, palette=None
        )
        ax_loc.set_title(f'Time {t_observed[n]} MSE {loss.item():.3f} Corr {spear:.3f}')
        ax_loc.set_xlabel(f'Observations')
        ax_loc.set_ylabel(f'Predictions')

def grid_visual(observations, predictions, t_observed, t_pred=None):
    if t_pred == None:
        t_pred = t_observed.clone()
        
    fig, axes = plt.subplots(observations.shape[1], observations.shape[2], figsize=(33, 20), sharex=True)
    obs = observations.cpu().numpy()
    pred = predictions.detach().cpu().numpy()

    for row in range(observations.shape[1]):
        for col in range(observations.shape[2]):
            if t_pred != None:
                x, y = t_pred.cpu().numpy(), pred[:, row, col]
                interpolation = interp1d(x, y, kind='quadratic')
                x = np.linspace(x.min(), x.max(), 100)
                y = interpolation(x)

                axes[row][col].plot(
                    t_pred.cpu().numpy(), 
                    pred[:, row, col], 
                    label='Predictions', 
                    color='#D15C6B',
                    marker='o',
                    fillstyle='none',
                    linestyle='',
                    markersize=5
                )

                axes[row][col].plot(
                    x, 
                    y, 
                    label='Quadratic Interpolation', 
                    color='#CDB3D4',
                    linestyle='-',
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
            axes[row][col].set_xticks(t_pred.cpu().numpy(), labels=t_pred.cpu().numpy().astype(int), rotation=45)
    
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.5)
    handles, labels = axes[row][col].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), fontsize='x-large')

def grid_visual_interpolate(observations, predictions, t_observed, t_pred=None):
    if t_pred == None:
        t_pred = t_observed.clone()
        
    fig, axes = plt.subplots(observations.shape[1], observations.shape[2], figsize=(33, 20), sharex=True)
    obs = observations.cpu().numpy()
    pred = predictions.detach().cpu().numpy()

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
    
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.5)
    handles, labels = axes[row][col].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), fontsize='x-large')
    plt.savefig('./demo.png', dpi=300, bbox_inches='tight')