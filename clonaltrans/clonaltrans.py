import torch
from torch import nn
from torchdiffeq import odeint_adjoint, odeint
from tqdm import tqdm
import numpy as np

from .ode_block import ODEBlock
from .utils import pbar_tb_description, input_data_form, tb_scalar

MSE = nn.MSELoss(reduction='mean')
SmoothL1 = nn.SmoothL1Loss(reduction='mean', beta=1.0)
GaussianNLL = nn.GaussianNLLLoss(reduction='mean', eps=1e-12)
MAE = nn.L1Loss(reduction='mean')

class CloneTranModel(nn.Module):
    def __init__(
        self,
        N, # (num_timpoints, num_clones, num_populations)
        L, # (num_populations, num_populations)
        config,
        writer: any = None,
        sample_N: any = None,
        extras: any = None
    ):
        super(CloneTranModel, self).__init__()
        self.N = N
        self.L = L
        self.D = config.D
        self.exponent = config.exponent

        self.input_N = torch.pow(N, exponent=self.exponent)
        # self.input_N = input_data_form(self.N, config.input_form, exponent=self.exponent)

        self.config = config
        self.writer = writer
        self.model_id = 0
        self.extras = extras

        self.init_ode_func()
        self.init_optimizer()
        self.get_masks()
        self.var_ub = torch.square(torch.maximum(
            torch.max(self.input_N, 0)[0].unsqueeze(0) / 7, 
            torch.tensor([1e-6]).to(config.gpu)
        ))

        if sample_N is None:
            print (f'\nData format: {config.input_form}. Exponent: {self.exponent}. Mean of input data: {self.input_N.mean().cpu():.3f}')
            print (f'{config.K_type}', '\n', self.ode_func)
            self.sample_N = torch.ones(self.N.shape).to(config.gpu)

        else:
            self.sample_N = sample_N

    def func(self, K_type):
        return ODEBlock(
            L=self.L,
            num_clones=self.N.shape[1],
            num_pops=self.N.shape[2],
            hidden_dim=self.config.hidden_dim, 
            activation=self.config.activation, 
            K_type=K_type,
            extras=self.extras
        ).to(self.config.gpu)

    def init_ode_func(self):
        self.ode_func = self.func(self.config.K_type)
        self.ode_func.supplement = [self.ode_func.supplement[i].to(self.config.gpu) for i in range(4)]

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.ode_func.parameters(), lr=self.config.learning_rate, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config.lrs_ms, gamma=0.5)

    def get_masks(self):
        self.used_L = self.L.clone()
        self.used_L.fill_diagonal_(1)
        self.used_L = self.used_L.unsqueeze(0)
        self.oppo_L = (self.used_L == 0).to(torch.float32) # (c, p, p)

        self.oppo_L_nondia = (self.L == 0).to(torch.float32).unsqueeze(0) # (1, p, p)

        self.zero_mask = (torch.sum(self.N, dim=0) == 0).to(torch.float32)

        self.no_proliferate = self.D.unsqueeze(0).to(torch.float32).to(self.config.gpu)

    def combine_K1_K2(self, K1, K2):
        for clone in range(K1.shape[0]):
            for pop in range(K1.shape[1]):
                K1[clone, pop, pop] += K2[clone, pop]
        return K1

    def get_Kt(self, tpoint, function):
        if tpoint == 0.0:
            tpoint += 0.01

        t_evaluation = torch.tensor([0.0, tpoint]).to(self.config.gpu)
        predictions = self.eval_model(t_evaluation)[1]
        K1_t, K2_t = function.get_K1_K2(predictions)
        return K1_t, K2_t
    
    def get_Kt_train(self, function):
        res_K1_t, res_K2_t = [], []

        for idx_time in range(self.input_N.shape[0]):
            K1_t, K2_t = function.get_K1_K2(self.input_N[idx_time])
            res_K1_t.append(K1_t)
            res_K2_t.append(K2_t)

        return torch.stack(res_K1_t), torch.stack(res_K2_t)
    
    def get_matrix_K(self, K_type='const', eval=False, tpoint=1.0):
        if K_type == 'const':
            # print (self.ode_func.K1.get_device(), self.ode_func.supplement[0].get_device(), self.ode_func.K2.get_device())
            K1 = torch.square(self.ode_func.K1 * self.ode_func.supplement[0] + self.ode_func.supplement[1]) * self.ode_func.K1_mask
            K2 = self.ode_func.K2.squeeze() * self.ode_func.supplement[2] + self.ode_func.supplement[3]
            K1, K2 = K1 / self.exponent, K2 / self.exponent

            if eval:
                return self.combine_K1_K2(K1, K2)
            else:
                return K1 * self.oppo_L_nondia, \
                    torch.cat([
                        torch.flatten((K1 > 10).to(torch.float32) * (K1 - 10)),
                        torch.flatten((K2 > 6).to(torch.float32) * (K2 - 6))
                    ]), \
                    torch.cat([
                        torch.flatten(K1 * self.zero_mask.unsqueeze(-1)), 
                        torch.flatten(K2 * self.zero_mask)
                    ]), \
                    torch.flatten((K2 * self.no_proliferate > 0).to(torch.float32) * (K2 * self.no_proliferate))     
               
        if K_type == 'dynamic':            
            if eval:
                K1_t, K2_t = self.get_Kt(tpoint, self.ode_func)
                matrix_K = self.combine_K1_K2(K1_t, K2_t)
                return matrix_K / self.exponent
            
            else:
                res_K1_t, res_K2_t = self.get_Kt_train(self.ode_func) # (t, c, p, p) and (t, c, p)
                res_K1_t, res_K2_t = res_K1_t / self.exponent, res_K2_t / self.exponent
                
                return res_K1_t * self.oppo_L_nondia.unsqueeze(0) / res_K2_t.shape[0], \
                    torch.cat([
                        torch.flatten((res_K1_t > 10).to(torch.float32) * (res_K1_t - 10)),
                        torch.flatten((res_K2_t > 6).to(torch.float32) * (res_K2_t - 6))
                    ]) / res_K2_t.shape[0], \
                    torch.cat([
                        torch.flatten(res_K1_t * (self.N == 0).unsqueeze(-1).to(torch.float32)), 
                        torch.flatten(res_K2_t * (self.N == 0).to(torch.float32))
                    ]) / res_K2_t.shape[0], \
                    torch.flatten((res_K2_t * self.no_proliferate.unsqueeze(0) > 0).to(torch.float32) * (res_K2_t * self.no_proliferate.unsqueeze(0))) / res_K2_t.shape[0]

    def compute_loss(self, y_pred, epoch, pbar):
        pena_nonL, pena_ub, pena_pop_zero, pena_k2_proli = self.get_matrix_K(K_type=self.config.K_type)
        thres = -1 if str(self.model_id).startswith('C') else int(self.config.num_epochs / 2)
        # thres = int(self.config.num_epochs / 2)

        if epoch == thres:
            for param in self.ode_func.parameters():
                param.requires_grad = False
            self.ode_func.std.requires_grad = True

        if epoch > thres:
            self.var = torch.broadcast_to(torch.square(self.ode_func.std), self.input_N.shape)[1:]
            loss_obs = GaussianNLL(self.input_N[1:] * self.sample_N[1:], y_pred[1:] * self.sample_N[1:], self.var)
        else:
            loss_obs = SmoothL1(y_pred[1:] * self.sample_N[1:], self.input_N[1:] * self.sample_N[1:])

        l1 = self.config.alpha * torch.sum(pena_nonL)   
        l2 = self.config.alpha * torch.sum(pena_ub)
        l3 = 0.05 * torch.linalg.norm(pena_k2_proli, ord=1)
        l4 = 0.01 * torch.linalg.norm(pena_pop_zero, ord=2)

        if epoch > thres:
            l5 = torch.flatten((self.var > self.var_ub).to(torch.float32) * (self.var - self.var_ub))
            l5 = 0.01 * torch.linalg.norm(l5, ord=1)
        else:
            l5 = torch.tensor([0.0]).to(self.config.gpu)

        tb_scalar(
            ['Model/LR', 'Model/NFEForward'],
            [self.optimizer.param_groups[0]['lr'], self.ode_func.nfe],
            epoch, self.writer
        )

        descrip = pbar_tb_description(
            ['ID', 'L/K2Pro', 'L/Upper', 'L/Recon', 'L/VarUb', 'L/Pop0'],
            [
                self.model_id, 
                torch.max(pena_k2_proli).item(), 
                l2.item() / self.config.alpha, 
                SmoothL1(y_pred * self.sample_N, self.input_N * self.sample_N).item(), 
                l5.item() / 0.01, 
                torch.max(pena_pop_zero).item()
            ],
            epoch, self.writer
        )
        pbar.set_description(descrip)
        self.ode_func.nfe = 0

        if epoch > thres:
            loss = loss_obs + l5
        else:
            loss = loss_obs + l1 + l2 + l4 + l3 + l5
        loss.backward()
        return loss

    def train_model(self, t_observed):
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

            if self.config.adjoint:
                y_pred = odeint_adjoint(self.ode_func, self.input_N[0], t_observed, 
                    method='dopri5', rtol=1e-4, atol=1e-4, options=dict(dtype=torch.float32))
            else:
                y_pred = odeint(self.ode_func, self.input_N[0], t_observed, 
                    method='dopri5', rtol=1e-4, atol=1e-4, options=dict(dtype=torch.float32))
                # y_pred = odeint(self.ode_func, self.input_N[0], t_observed, 
                #     method='rk4', options=dict(step_size=0.1))

            loss = self.compute_loss(y_pred, epoch, pbar)

            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def eval_model(self, t_eval):
        self.ode_func.eval()
        # self.variance = torch.square(self.ode_func.std)

        if self.config.adjoint:
            y_pred = odeint_adjoint(self.ode_func, self.input_N[0], t_eval, method='dopri5')
        else:
            y_pred = odeint(self.ode_func, self.input_N[0], t_eval, method='dopri5')
        
        return y_pred
