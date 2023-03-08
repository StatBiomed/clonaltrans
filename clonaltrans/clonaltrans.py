import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint
# from torchdiffeq import odeint

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
MSE = nn.MSELoss(reduction='mean')
SmoothL1 = nn.SmoothL1Loss(reduction='mean', beta=1.0)

def activation_helper(activation):
    if activation == 'gelu':
        act = nn.GELU()
    if activation == 'sigmoid':
        act = nn.Sigmoid()
    if activation == 'tanh':
        act = nn.Tanh()
    if activation == 'relu':
        act = nn.ReLU()
    if activation == 'leakyrelu':
        act = nn.LeakyReLU()
    if activation is None:
        def act(x):
            return x
    return act

class ODEBlock(nn.Module):
    def __init__(self, N, input_dim, hidden_dim, activation: str = 'gelu'):
        super(ODEBlock, self).__init__()
        self.N = N
        self.activation = activation_helper(activation)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()

        for i in range(N.shape[1]):
            self.encode.append(nn.Linear(input_dim, hidden_dim))
            self.decode.append(nn.Linear(hidden_dim, input_dim))

    def forward(self, t, y):
        outputs = []

        for i in range(self.N.shape[1]):
            z = self.encode[i](y[i])
            z = self.activation(z)
            z = self.decode[i](z)
            outputs.append(z)

        return torch.stack(outputs)

    def forward_direct(self, t, y):
        # Calculate K_influx and K_outflux matrices (m, p, p)
        # K_influx should be the transpose of K_outflux in dim 1 and 2
        K_base = self.K[-1].unsqueeze(0)
        K_outflux = torch.cat([self.K[:-1] + K_base, K_base], dim=0)
        # Calculate dydt for each clone in range [0, n-1]
        y = y.unsqueeze(1)
        dydt = torch.bmm(y, K_outflux.transpose(1, 2)) - torch.bmm(y, K_outflux)

        return dydt.squeeze() # (m, p)

class CloneTranModel(nn.Module):
    def __init__(
        self,
        N, # (num_timpoints, num_clones, num_populations)
        L, # (num_populations, num_populations)
        config,
        comment: str = ""
    ):
        super(CloneTranModel, self).__init__()
        self.N = N
        self.L = L
        self.log_N = torch.log(N + 1e-6)

        self.config = config
        self.writer = SummaryWriter(log_dir=None, comment=comment)

        self.ode_func = ODEBlock(N=N, input_dim=N.shape[2], hidden_dim=16, activation=config.activation)
        self.init_optimizer()

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            [
                {'params': self.ode_func.parameters()}
            ],
            lr=self.config.learning_rate,
            amsgrad=True
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=300, gamma=0.5)

    def get_matrix_K(self):
        #* matrix_K (num_clones, num_populations, num_populations)
        #* matrix_K[-1] = base K(1) for background cells
        #* matrix_K[:-1] = parameter delta for each meta-clone specified in paper
        matrix_K = []
        for i in range(self.N.shape[1] - 1):
            matrix_K.append(torch.matmul(self.ode_func.encode[i].weight.T, self.ode_func.decode[i].weight.T))
        matrix_K.append(torch.matmul(self.ode_func.encode[-1].weight.T, self.ode_func.decode[-1].weight.T) * self.L)
        return torch.stack(matrix_K)

    def compute_loss(self, y_pred, epoch, pbar):
        self.matrix_K = self.get_matrix_K()
        loss_obs = SmoothL1(y_pred, self.log_N)
        loss_K = self.config.beta * torch.linalg.norm(self.matrix_K[-1], ord='fro')
        loss_delta = self.config.alpha * torch.sum(torch.abs(self.matrix_K[:-1]))

        pbar.set_description(f"Delta {loss_delta.item():.3f}, BaseK {loss_K.item():.3f}, Observation {loss_obs.item():.3f}")
        self.tb_writer(epoch, loss_delta, loss_K, loss_obs)
        return loss_obs + loss_K + loss_delta

    def train_model(self, t_observed):
        self.ode_func.train()
        # self.nonzero = (self.N != 0).type(torch.float32)

        pbar = tqdm(range(self.config.num_epochs))
        for epoch in pbar:
            self.optimizer.zero_grad()

            y_pred = odeint(self.ode_func, self.log_N[0], t_observed, method='dopri5')
            loss = self.compute_loss(y_pred, epoch, pbar)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def eval_model(self, t_eval):
        self.ode_func.eval()

        y_pred = odeint(self.ode_func, self.log_N[0], t_eval, method='dopri5')
        return torch.exp(y_pred)

    def tb_writer(
        self,
        iter,
        loss_delta,
        loss_K,
        loss_obs
    ):
        self.writer.add_scalar('Loss/Delta', loss_delta.item(), iter)
        self.writer.add_scalar('Loss/K', loss_K.item(), iter)
        self.writer.add_scalar('Loss/Observation', loss_obs.item(), iter)
        self.writer.add_scalar('Loss/Learing_Rate', self.optimizer.param_groups[0]['lr'], iter)
