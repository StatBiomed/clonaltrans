import torch
from torch import nn
from torchdiffeq import odeint_adjoint, odeint
from tqdm import tqdm

from .ode_block import ODEBlock
from .utils import timeit, pbar_tb_description, input_data_form, tb_scalar

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
        writer
    ):
        super(CloneTranModel, self).__init__()
        self.N = N
        self.L = L
        self.exponent = config.exponent

        self.input_N = input_data_form(N, config.input_form, exponent=self.exponent)
        print (f'\nData format: {config.input_form}. Exponent: {self.exponent}. Mean of input data: {self.input_N.mean().cpu():.3f}')

        self.config = config
        self.writer = writer

        self.ode_func = ODEBlock(
            num_tpoints=N.shape[0],
            num_clones=N.shape[1],
            num_pops=N.shape[2],
            hidden_dim=config.hidden_dim, 
            activation=config.activation, 
            K_type=config.K_type,
            lam=config.lam
        )
        print (self.ode_func)
        
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
        self.mask = (self.N.sum(0) != 0).to(torch.float32) # (c, p)

        self.used_L = self.L.clone()
        self.used_L.fill_diagonal_(1)
        self.used_L = self.used_L.unsqueeze(0)
        self.oppo_L = (self.used_L == 0).to(torch.float32) # (c, p, p)

        self.oppo_L_nondia = (self.L == 0).to(torch.float32).unsqueeze(0) # (1, p, p)

    def combine_K1_K2(self, K1, K2):
        for clone in range(K1.shape[0]):
            for pop in range(K1.shape[1]):
                K1[clone, pop, pop] += K2[clone, pop]
        return K1

    def get_Kt(self, tpoint):
        if tpoint == 0.0:
            tpoint += 0.01

        t_evaluation = torch.tensor([0.0, tpoint]).to(self.config.gpu)
        predictions = self.eval_model(t_evaluation)[1]
        K1_t, K2_t = self.ode_func.get_K1_K2(predictions)
        return K1_t, K2_t
    
    def get_Kt_train(self):
        res_K1_t, res_K2_t = [], []

        for idx_time in range(self.input_N.shape[0]):
            K1_t, K2_t = self.ode_func.get_K1_K2(self.input_N[idx_time])
            res_K1_t.append(K1_t)
            res_K2_t.append(K2_t)

        return torch.stack(res_K1_t), torch.stack(res_K2_t)
    
    def get_matrix_K(self, eval=False, tpoint=1.0, sep='mixture'):
        if self.config.K_type == 'const':
            K1 = torch.square(self.ode_func.K1) * self.ode_func.K1_mask

            if eval:
                matrix_K = self.combine_K1_K2(K1, self.ode_func.K2.squeeze())
                return matrix_K / self.exponent
            else:
                return K1 / self.exponent * self.oppo_L_nondia, \
                    ((self.ode_func.K2.squeeze() / self.exponent - 6) > 0).to(torch.float32) * (self.ode_func.K2.squeeze() / self.exponent - 6), \
                    torch.tensor([0.]), \
                    torch.tensor([0.])
        
        if self.config.K_type == 'dynamic':            
            if eval:
                K1_t, K2_t = self.get_Kt(tpoint)
                matrix_K = self.combine_K1_K2(K1_t, K2_t)
                return matrix_K / self.exponent
            
            else:
                res_K1_t, res_K2_t = self.get_Kt_train() # (t, c, p, p) and (t, c, p)
                return res_K1_t / self.exponent * self.oppo_L_nondia.unsqueeze(0), \
                    ((res_K2_t / self.exponent - 6) > 0).to(torch.float32) * (res_K2_t / self.exponent - 6), \
                    torch.tensor([0.]), \
                    torch.tensor([0.])
        
        if self.config.K_type == 'mixture':
            K1 = torch.square(self.ode_func.K1) * self.ode_func.K1_mask
            K2 = self.ode_func.K2.squeeze()

            if eval:
                assert sep in ['const', 'dynamic', 'mixture']
                K1_t, K2_t = self.get_Kt(tpoint)

                if sep == 'const':
                    return self.combine_K1_K2(K1, K2) / self.exponent
                if sep == 'dynamic':
                    return self.combine_K1_K2(K1_t, K2_t) / self.exponent
                if sep == 'mixture':
                    return self.combine_K1_K2(K1 + K1_t, K2 + K2_t) / self.exponent
            
            else:
                K1, K2 = K1.unsqueeze(0), K2.unsqueeze(0)
                res_K1_t, res_K2_t = self.get_Kt_train() # (t, c, p, p) and (t, c, p)

                return torch.concat([K1, res_K1_t]) / self.exponent * self.oppo_L_nondia.unsqueeze(0), \
                    ((torch.concat([K2, res_K2_t]) / self.exponent - 6) > 0).to(torch.float32) * (torch.concat([K2, res_K2_t]) / self.exponent - 6), \
                    torch.concat([torch.flatten(self.ode_func.K1_encode), torch.flatten(self.ode_func.K1_decode)]), \
                    torch.concat([torch.flatten(self.ode_func.K2_encode), torch.flatten(self.ode_func.K2_decode)])
                    # res_K1_t / self.exponent * self.L.unsqueeze(0).unsqueeze(0), \
                    # res_K2_t / self.exponent

    def compute_loss(self, y_pred, epoch, pbar):
        pena_K1_nonL, pena_K2_ub, pena_K1_t, pena_K2_t = self.get_matrix_K()

        if self.config.include_var:
            var = torch.broadcast_to(torch.square(self.ode_func.std), self.input_N.shape)
            loss_obs = GaussianNLL(self.input_N, y_pred, var)
        else:
            loss_obs = SmoothL1(y_pred, self.input_N)

        l1 = self.config.alpha * torch.sum(pena_K1_nonL)   
        l2 = self.config.alpha * torch.sum(pena_K2_ub)
        l3 = self.config.beta * (torch.sum(torch.abs(pena_K1_t)) + torch.sum(torch.abs(pena_K2_t)))

        tb_scalar(
            ['Model/LR', 'Model/NFEForward'],
            [self.optimizer.param_groups[0]['lr'], self.ode_func.nfe],
            epoch, self.writer
        )

        descrip = pbar_tb_description(
            ['L/NonL', 'L/K2Ub', 'L/Recon', 'L/K(t)L', 'Model/Lam'],
            [l1.item() / self.config.alpha, l2.item() / self.config.alpha, loss_obs.item(), l3.item() / (self.config.beta + 1e-6), self.ode_func.lam.item()],
            epoch, self.writer
        )
        pbar.set_description(descrip)
        self.ode_func.nfe = 0

        loss = loss_obs + l1 + l2 + l3
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

            loss = self.compute_loss(y_pred, epoch, pbar)

            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def eval_model(self, t_eval, log_output=False):
        self.ode_func.eval()
        self.variance = torch.square(self.ode_func.std)

        if self.config.adjoint:
            y_pred = odeint_adjoint(self.ode_func, self.input_N[0], t_eval, method='dopri5')
        else:
            y_pred = odeint(self.ode_func, self.input_N[0], t_eval, method='dopri5')
        
        if self.config.input_form in ['raw', 'shrink', 'root']:
            return y_pred
        
        elif self.config.input_form == 'log' and log_output:
            return y_pred
        
        else: return (torch.exp(y_pred) - 1.0)
