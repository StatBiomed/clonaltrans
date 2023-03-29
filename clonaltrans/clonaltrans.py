import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint
from .ode_block import ODEBlock

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import time
import inspect

MSE = nn.MSELoss(reduction='mean')
SmoothL1 = nn.SmoothL1Loss(reduction='mean', beta=1.0)

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

        self.input_N = self.log_N if config.log_data else N

        self.config = config
        self.writer = writer

        self.ode_func = ODEBlock(N=N, input_dim=N.shape[2], hidden_dim=config.hidden_dim, activation=config.activation)
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
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.config.lrs_step, 
            gamma=self.config.lrs_gamma
        )

    def get_masks(self):
        self.mask = (self.N.sum(0) != 0).to(torch.float32).unsqueeze(0)

    def get_matrix_K(self):
        #* matrix_K (num_clones, num_populations, num_populations)
        #* matrix_K[-1] = base K(1) for background cells
        #* matrix_K[:-1] = parameter delta for each meta-clone specified in paper
        matrix_K = []
        for i in range(self.N.shape[1] - 1):
            matrix_K.append(torch.matmul(self.ode_func.encode[i].weight.T, self.ode_func.decode[i].weight.T) * self.L)
        matrix_K.append(torch.matmul(self.ode_func.encode[-1].weight.T, self.ode_func.decode[-1].weight.T) * self.L)
        return torch.stack(matrix_K)

    def compute_loss(self, y_pred, epoch, pbar):
        self.matrix_K = self.get_matrix_K()
        loss_obs = SmoothL1(y_pred * self.mask, self.input_N * self.mask)
        loss_K = self.config.beta * torch.linalg.norm(self.matrix_K[-1], ord='fro')
        loss_delta = self.config.alpha * torch.sum(torch.abs(self.matrix_K[:-1]))

        pbar.set_description(f"Delta {loss_delta.item():.3f}, BaseK {loss_K.item():.3f}, Observation {loss_obs.item():.3f}")
        self.tb_writer(epoch, loss_delta, loss_K, loss_obs)

        loss = loss_obs + loss_K + loss_delta
        loss.backward()
        return loss

    def timeit(self, func, epoch):
        def wrapper(*args, **kwargs):
            start_time = time.monotonic()
            result = func(*args, **kwargs)
            end_time = time.monotonic()

            # Use inspect to get the line number of the decorated function call
            line_number = inspect.currentframe().f_back.f_lineno

            # print(f"Line {line_number} in {func.__name__} took {end_time - start_time:.6f} seconds")
            self.writer.add_scalar(f'Time/{func.__name__}', np.round(end_time - start_time, 3), epoch)
            
            return result
        return wrapper

    def train_model(self, t_observed):
        self.ode_func.train()

        pbar = tqdm(range(self.config.num_epochs))
        for epoch in pbar:
            self.optimizer.zero_grad()

            if self.config.inspect:
                y_pred = self.timeit(odeint, epoch)(self.ode_func, self.input_N[0], t_observed, method='dopri5')
                loss = self.timeit(self.compute_loss, epoch)(y_pred, epoch, pbar)
            else:
                y_pred = odeint(self.ode_func, self.input_N[0], t_observed, method='dopri5')
                loss = self.compute_loss(y_pred, epoch, pbar)

            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def eval_model(self, t_eval):
        self.ode_func.eval()

        y_pred = odeint(self.ode_func, self.input_N[0], t_eval, method='dopri5')
        return (torch.exp(y_pred) - 1e-6) * self.mask if self.config.log_data else y_pred * self.mask

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
