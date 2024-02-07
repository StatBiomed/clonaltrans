import torch
import pandas as pd
import os
import numpy as np

class ClonalTransDataLoader():
    def __init__(
        self,
        data_dir,
        num_populations,
        logger,
        annots,
        graphs,
        day_zero,
        cell_counts
    ):
        self.data_dir = data_dir
        self.num_pop = num_populations
        self.logger = logger

        self.annotations = annots
        self.trans_graph = graphs
        self.day_zero = day_zero
        self.cell_counts = cell_counts

        self.paga, self.array_total = self.read_datasets()
        self.logger.info(f'Mean of input data: {self.array_total.mean().cpu():.3f}\n')

    def read_datasets(self):
        # PAGA Topology (Population Adjacency Matrix), same for each clones
        paga = pd.read_csv(os.path.join(self.data_dir, self.trans_graph), index_col=0).astype(np.int32)
        self.logger.info(f'Topology graph loaded with shape {paga.shape}.')

        # number of cells (per timepoints, per meta-clone, per population)
        array = np.loadtxt(os.path.join(self.data_dir, self.cell_counts))
        array_ori = array.reshape(array.shape[0], int(array.shape[1] // self.num_pop), self.num_pop)
        array_ori = torch.swapaxes(torch.tensor(array_ori, dtype=torch.float32), 0, 1)
        self.logger.info(f'Input cell data (num_timepoints {array_ori.shape[0]}, num_clones {array_ori.shape[1]}, num_populations {array_ori.shape[2]}) loaded.')

        # init HSCs in Day 0
        init_con = pd.read_csv(os.path.join(self.data_dir, self.day_zero), index_col=0).astype(np.float32)
        day_zero = np.zeros((array_ori.shape[1], array_ori.shape[2]))
        day_zero[:, 0] = init_con['leiden'].values

        array_ori = torch.cat((torch.tensor(day_zero, dtype=torch.float32).unsqueeze(0), array_ori), axis=0)
        self.logger.info(f'Day 0 has been added. Input data shape: {array_ori.shape}')

        return torch.tensor(paga.values, dtype=torch.float32), array_ori

    def get_datasets(self):
        return self.array_total, self.paga