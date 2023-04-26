import torch
from torch import nn
from torchdiffeq import odeint_adjoint, odeint
from tqdm import tqdm

from .ode_block import ODEBlock
from .utils import timeit, pbar_tb_description, input_data_form

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
        self.atol = 1e-1
        self.exponent = 1 / 4

        self.input_N = input_data_form(N, config.input_form, exponent=self.exponent)
        print (f'\nData format: {config.input_form}. Mean of input data: {self.input_N.mean().cpu():.3f}')

        self.config = config
        self.writer = writer

        self.ode_func = ODEBlock(
            N=N, 
            L=torch.broadcast_to(L.unsqueeze(0), (N.shape[1], N.shape[2], N.shape[2])),
            hidden_dim=config.hidden_dim, 
            activation=config.activation, 
            config=config
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
        self.mask = (self.N.sum(0) != 0).to(torch.float32).unsqueeze(0)

        self.oppo_L = self.L.clone()
        self.oppo_L.fill_diagonal_(1)
        self.oppo_L = torch.broadcast_to(self.oppo_L.unsqueeze(0), (self.N.shape[1], self.N.shape[2], self.N.shape[2]))
        self.oppo_L = (self.oppo_L == 0).to(torch.float32)

        self.used_L = (self.oppo_L == 0).to(torch.float32)

    def get_matrix_K(self, eval=False):
        '''
        matrix_K (num_clones, num_populations, num_populations)
        matrix_K[-1] = base K(1) for background cells
        matrix_K[:-1] = parameter delta for each meta-clone specified in paper
        '''
        
        if self.config.num_layers == 1:
            if eval:
                matrix_K = torch.square(self.ode_func.K1) * self.ode_func.L

                for clone in range(matrix_K.shape[0]):
                    for pop in range(matrix_K.shape[1]):
                        matrix_K[clone, pop, pop] += self.ode_func.K2[clone, pop]
                
                return matrix_K
            else:
                return torch.zeros((self.L.shape))
        
        if self.config.num_layers == 2:
            matrix_K = torch.bmm(self.ode_func.encode, self.ode_func.decode)

            if eval:
                return matrix_K
            else:
                return matrix_K * self.oppo_L, \
                    ((matrix_K * self.L) < 0).to(torch.float32) * matrix_K, \
                    (((matrix_K * 4 - 6.) * self.used_L) > 0).to(torch.float32) * (matrix_K * 4 - 6.)

    def compute_loss(self, y_pred, epoch, pbar, y_pred_eval=None):
        self.matrix_K, off_diagonal, upper_bound = self.get_matrix_K()

        loss_obs = SmoothL1(y_pred * self.mask, self.input_N * self.mask)

        loss_K = self.config.beta * torch.sum(torch.abs(self.matrix_K[-1]))
        loss_delta = self.config.alpha * torch.sum(torch.abs(self.matrix_K[:-1]))    
        loss_offdia = 0.01 * torch.sum(off_diagonal)
        loss_upper = 0.01 * torch.sum(upper_bound)
        
        descrip = pbar_tb_description(
            ['Loss/Delta', 'Loss/K', 'Loss/Obs', 'Loss/LR', 'NFE/Forward', 'Loss/OffDia', 'Loss/Upper'],
            [loss_delta.item(), loss_K.item(), loss_obs.item(), self.optimizer.param_groups[0]['lr'], self.ode_func.nfe, loss_offdia.item(), loss_upper.item()],
            epoch, self.writer
        )
        pbar.set_description(descrip)
        self.ode_func.nfe = 0

        loss = loss_obs + loss_K + loss_delta - loss_offdia + loss_upper
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
    def eval_model(self, t_eval, log_output=False):
        self.ode_func.eval()

        if self.config.adjoint:
            y_pred = odeint_adjoint(self.ode_func, self.input_N[0], t_eval, method='dopri5')
        else:
            y_pred = odeint(self.ode_func, self.input_N[0], t_eval, method='dopri5')
        
        if self.config.input_form in ['raw', 'shrink', 'root']:
            return y_pred * self.mask
        
        elif self.config.input_form == 'log' and log_output:
            self.mask_log = torch.broadcast_to((self.mask == 0), y_pred.shape)
            y_pred[self.mask_log] = torch.log(torch.tensor(self.atol))
            return y_pred
        
        else: return (torch.exp(y_pred) - self.atol) * self.mask
