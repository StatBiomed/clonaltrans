import torch
from torch import nn
from torchdiffeq import odeint_adjoint, odeint
from tqdm import tqdm

from .ode_block import ODEBlock
from .utils import pbar_tb_description, input_data_form, tb_scalar

MSE = nn.MSELoss(reduction='mean')
SmoothL1 = nn.SmoothL1Loss(reduction='mean', beta=1.0)
GaussianNLL = nn.GaussianNLLLoss(reduction='mean')
MAE = nn.L1Loss(reduction='mean')

class CloneTranModel(nn.Module):
    def __init__(
        self,
        N, # (num_timpoints, num_clones, num_populations)
        L, # (num_populations, num_populations)
        config,
        writer: any = None,
        sample_N: any = None
    ):
        super(CloneTranModel, self).__init__()
        self.N = N
        self.L = L
        self.exponent = config.exponent
        self.input_N = input_data_form(self.N, config.input_form, exponent=self.exponent)

        self.config = config
        self.writer = writer
        self.model_id = 0

        self.init_ode_func()
        self.init_optimizer()
        self.get_masks()
        self.var_ub = torch.square(torch.maximum(
            torch.max(self.input_N, 0)[0].unsqueeze(0) / 10, 
            torch.tensor([1e-2]).to(config.gpu)
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
            K_type=K_type
        )

    def get_fractions(self):
        self.frac = self.N / torch.sum(self.N, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)

    def init_ode_func(self):
        self.ode_func = self.func(self.config.K_type)
        self.ode_func_dyna = self.func('dynamic')
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

            if eval:
                matrix_K = self.combine_K1_K2(K1, K2)
                return matrix_K / self.exponent
            else:
                return K1 / self.exponent * self.oppo_L_nondia, \
                    torch.concat([
                        torch.flatten((K1 / self.exponent > 10).to(torch.float32) * (K1 / self.exponent - 10)),
                        torch.flatten((K2 / self.exponent > 6).to(torch.float32) * (K2 / self.exponent - 6))
                    ]), \
                    torch.concat([
                        torch.flatten(K1 / self.exponent * self.zero_mask.unsqueeze(-1)), 
                        torch.flatten(K2 * self.zero_mask / self.exponent)
                    ]), \
                    torch.tensor([0.])
        
        if K_type == 'dynamic':            
            if eval:
                K1_t, K2_t = self.get_Kt(tpoint, self.ode_func)
                matrix_K = self.combine_K1_K2(K1_t, K2_t)
                return matrix_K / self.exponent
            
            else:
                res_K1_t, res_K2_t = self.get_Kt_train(self.ode_func) # (t, c, p, p) and (t, c, p)
                return res_K1_t / self.exponent * self.oppo_L_nondia.unsqueeze(0), \
                    torch.concat([
                        torch.flatten((res_K1_t / self.exponent > 10).to(torch.float32) * (res_K1_t / self.exponent - 10)),
                        torch.flatten((res_K2_t / self.exponent > 6).to(torch.float32) * (res_K2_t / self.exponent - 6))
                    ]), \
                    torch.concat([
                        torch.flatten(res_K1_t / self.exponent * (self.N == 0).unsqueeze(-1).to(torch.float32)), 
                        torch.flatten(res_K2_t / self.exponent * (self.N == 0).to(torch.float32))
                    ]), \
                    torch.tensor([0.])

    def compute_loss(self, y_pred, epoch, pbar):
        pena_nonL, pena_ub, pena_pop_zero, pena_K1K2_para = self.get_matrix_K(K_type=self.config.K_type)

        if self.config.include_var:
            var = torch.broadcast_to(torch.square(self.ode_func.std), self.input_N.shape)
            loss_obs = GaussianNLL(self.input_N * self.sample_N, y_pred * self.sample_N, var)
        else:
            loss_obs = SmoothL1(y_pred * self.sample_N, self.input_N * self.sample_N)

        l1 = self.config.alpha * torch.sum(pena_nonL)   
        l2 = self.config.alpha * torch.sum(pena_ub)
        # l3 = self.config.beta * torch.sum(torch.abs(pena_K1K2_para))
        l4 = 0.01 * torch.sum(torch.linalg.vector_norm(pena_pop_zero, ord=2))

        if self.config.include_var:
            l5 = 0.01 * torch.sum((var > self.var_ub).to(torch.float32) * (var - self.var_ub))
        else:
            l5 = torch.tensor([0.0]).to(self.config.gpu)

        tb_scalar(
            ['Model/LR', 'Model/NFEForward'],
            [self.optimizer.param_groups[0]['lr'], self.ode_func.nfe],
            epoch, self.writer
        )

        descrip = pbar_tb_description(
            ['ID', 'L/NonL', 'L/K2Ub', 'L/Recon', 'L/VarUb', 'L/L2Norm'],
            [
                self.model_id, 
                l1.item() / self.config.alpha, 
                l2.item() / self.config.alpha, 
                SmoothL1(y_pred * self.sample_N, self.input_N * self.sample_N).item(), 
                l5.item() / 0.01, 
                l4.item() / 0.01
            ],
            epoch, self.writer
        )
        pbar.set_description(descrip)
        self.ode_func.nfe = 0

        loss = loss_obs + l1 + l2 + l5 + l4
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
        self.ode_func_dyna.train()

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
        self.variance = torch.square(self.ode_func.std)

        if self.config.adjoint:
            y_pred = odeint_adjoint(self.ode_func, self.input_N[0], t_eval, method='dopri5')
        else:
            y_pred = odeint(self.ode_func, self.input_N[0], t_eval, method='dopri5')
        
        return y_pred
