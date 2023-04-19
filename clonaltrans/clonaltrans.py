import torch
from torch import nn
from torchdiffeq import odeint_adjoint, odeint
from .ode_block import ODEBlock

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from .utils import timeit, pbar_tb_description

MSE = nn.MSELoss(reduction='mean')
SmoothL1 = nn.SmoothL1Loss(reduction='mean', beta=1.0)
MAE = nn.L1Loss(reduction='mean')

class CloneTranModel(nn.Module):
    def __init__(
        self,
        N, # (num_timpoints, num_clones, num_populations)
        L, # (num_populations, num_populations)
        config,
        writer
    ):
        super(CloneTranModel, self).__init__()
        self.N = N
        self.L = L
        self.log_N = torch.log(N + 1e-6)

        # self.input_N = self.log_N if config.log_data else N
        self.input_N = N
        print (f'Mean of original data: {N.mean().cpu():.3f}')
        print (f'Mean of input data: {self.input_N.mean().cpu():.3f}')

        self.config = config
        self.writer = writer

        self.ode_func = ODEBlock(
            N=N, 
            L=L,
            hidden_dim=16, 
            activation=config.activation, 
            config=config
        )
        
        self.init_optimizer()
        self.get_masks()

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            [
                {'params': self.ode_func.parameters()}
            ],
            lr=self.config.learning_rate,
            amsgrad=True
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, 
            milestones=self.config.lrs_ms, 
            gamma=self.config.lrs_gamma
        )

    def get_masks(self):
        self.mask = (self.N.sum(0) != 0).to(torch.float32).unsqueeze(0)

    def get_matrix_K(self):
        #* matrix_K (num_clones, num_populations, num_populations)
        #* matrix_K[-1] = base K(1) for background cells
        #* matrix_K[:-1] = parameter delta for each meta-clone specified in paper
        
        if self.config.num_layers == 1:
            matrix_K = torch.square(self.ode_func.K1) * self.ode_func.L

            for clone in range(matrix_K.shape[0]):
                for pop in range(matrix_K.shape[1]):
                    matrix_K[clone, pop, pop] += self.ode_func.K2[clone, pop]
        
        if self.config.num_layers == 2:
            matrix_K = torch.bmm(self.ode_func.encode, self.ode_func.decode) * self.ode_func.L
        
            # matrix_K = []
            # for i in range(self.N.shape[1]):
            #     matrix_K.append(
            #         torch.matmul(
            #             self.ode_func.encode[i].weight.T, 
            #             self.ode_func.decode[i].weight.T
            #         ) * self.L)
            # return torch.stack(matrix_K)

        return matrix_K

    def compute_loss(self, y_pred, epoch, pbar, y_pred_eval=None):
        self.matrix_K = self.get_matrix_K()
        # loss_obs = SmoothL1(torch.log(y_pred + 1e-6) * self.mask, self.log_N * self.mask)
        loss_obs = SmoothL1(y_pred * self.mask, self.input_N * self.mask)

        loss_K = self.config.beta * torch.linalg.norm(self.matrix_K[-1], ord='fro')
        loss_delta = self.config.alpha * torch.sum(torch.abs(self.matrix_K[:-1]))        
        
        descrip = pbar_tb_description(
            ['Loss/Delta', 'Loss/K', 'Loss/Obs', 'Loss/LR', 'NFE/Forward'],
            [loss_delta.item(), loss_K.item(), loss_obs.item(), self.optimizer.param_groups[0]['lr'], self.ode_func.nfe],
            epoch, self.writer
        )
        pbar.set_description(descrip)
        self.ode_func.nfe = 0

        loss = loss_obs + loss_K + loss_delta
        loss.backward()
        return loss

    def train_model(self, t_observed, t_eval=None):
        '''
        For most problems, good choices are the default dopri5, 
        or to use rk4 with options=dict(step_size=...) set appropriately small. 
        Adjusting the tolerances (adaptive solvers) or step size (fixed solvers), 
        will allow for trade-offs between speed and accuracy.
        '''

        self.ode_func.train()

        pbar = tqdm(range(self.config.num_epochs))
        for epoch in pbar:
            self.optimizer.zero_grad()

            # for p in self.ode_func.K1.parameters():
            #     p.data.clamp_(0)

            y_pred = timeit(
                odeint_adjoint if self.config.adjoint else odeint, epoch, self.writer
            )(
                self.ode_func, self.input_N[0], t_observed, method='dopri5',
                rtol=1e-5, options=dict(dtype=torch.float32),
            )

            loss = timeit(self.compute_loss, epoch, self.writer)(y_pred, epoch, pbar)

            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def eval_model(self, t_eval):
        self.ode_func.eval()

        if self.config.adjoint:
            y_pred = odeint_adjoint(self.ode_func, self.input_N[0], t_eval, method='dopri5')
        else:
            y_pred = odeint(self.ode_func, self.input_N[0], t_eval, method='dopri5')
        
        # return (torch.exp(y_pred) - 1e-6) * self.mask if self.config.log_data else y_pred * self.mask
        return y_pred * self.mask
