import pandas as pd
import os
import torch
import numpy as np

def get_topo_obs(
    data_dir, 
    fill_diagonal: bool = True, 
    device: int = 0
):
    # PAGA Topology (Population Adjacency Matrix), same for each clones?
    paga = pd.read_csv(os.path.join(data_dir, 'graph_table.csv'), index_col=0).astype(np.int32)
    print ('Topology graph loaded.')

    # number of cells (per timepoints, per meta-clone, per population)
    array = np.loadtxt(os.path.join(data_dir, 'kinetics_array_correction_factor.txt'))
    array_ori = array.reshape(array.shape[0], array.shape[1] // 11, 11)
    array_ori = torch.swapaxes(torch.tensor(array_ori, dtype=torch.float32), 0, 1)
    print ('Input cell data (num_timepoints, num_clones, num_populations) loaded.')

    # generate background cells
    background = torch.sum(array_ori, axis=1).unsqueeze(1)
    array_total = torch.concatenate((array_ori, background), axis=1)
    print ('Background reference cells generated.')

    if fill_diagonal:
        np.fill_diagonal(paga.values, 1)

    # some simulation data on cell counts
    # array_total[1] = array_total[0] * 10 + torch.normal(torch.zeros(array_total[0].shape), torch.ones(array_total[0].shape)) * 10
    # array_total[1][array_total[1] < 0] = 0
    # array_total[2] = array_total[1] + 50 + torch.normal(torch.zeros(array_total[1].shape), torch.ones(array_total[1].shape)) * 10
    # array_total[2][array_total[2] < 0] = 0
    # array_total[2] = torch.where(array_total[2] > array_total[1], array_total[2] / 100, array_total[2])
    # array_total[2][array_total[2] < 0] = 0

    return torch.tensor(paga.values, dtype=torch.float32, device=device), array_total.to(device)