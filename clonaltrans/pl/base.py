from torch import nn
import math
import numpy as np
import pandas as pd
import os
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

def transit_K(model, K, index=None, columns=None):
    anno = pd.read_csv(os.path.join(
        model.config['data_loader']['args']['data_dir'], 
        model.config['data_loader']['args']['annots']
    ))

    return pd.DataFrame(
        index=anno['populations'].values[:model.N.shape[2]] if index is None else index, 
        columns=anno['populations'].values[:model.N.shape[2]] if columns is None else columns, 
        data=K
    )

def interpolate_1d(tpoints, y, kind='linear'):
    f = interp1d(np.linspace(0, int(tpoints[-1]), int(tpoints[-1]) + 1), y, kind=kind)
    newx = np.linspace(0, int(tpoints[-1]), 50)
    newy = f(newx)
    return newx, newy
