import torch
from torch import nn
from torchdiffeq import odeint_adjoint, odeint
from tqdm import tqdm

from utils import tb_scalar, pbar_tb_description

GaussianNLL = nn.GaussianNLLLoss(reduction='mean', eps=1e-12)
PoissonNLL = nn.PoissonNLLLoss(log_input=False, reduction='mean')

class CloneTranModel(nn.Module):
    def __init__(
        self,
        N, # (num_timpoints, num_clones, num_populations)
        L, # (num_populations, num_populations)
        config,
        writer,
        model,
        optimizer,
        scheduler,
        t_observed,
        sample_N: any = None,
        extras: any = None
    ):
        super(CloneTranModel, self).__init__()
        self.N = N
        self.L = L
        self.D = torch.tensor(config['user_trainer']['no_proliferation_pops'])
        self.model = model
        self.exponent = config['data_loader']['args']['exponent']

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.t_observed = t_observed
        self.logger = config['data_loader']['args']['logger']

        self.config = config
        self.writer = writer
        self.model_id = 0
        self.extras = extras

        self.get_masks()

        self.var_ub = torch.square(torch.maximum(
            torch.max(self.N, 0)[0].unsqueeze(0) / 7, 
            torch.tensor([1e-6]).to(self.config['gpu_id'])
        ))

        self.sample_N = torch.ones(self.N.shape).to(self.config['gpu_id']) if sample_N is None else sample_N

    def get_masks(self):
        self.used_L = self.L.clone()
        self.used_L.fill_diagonal_(1)
        self.used_L = self.used_L.unsqueeze(0)

        self.oppo_L = (self.used_L == 0).to(torch.float32) # (c, p, p)
        self.oppo_L_nondia = (self.L == 0).to(torch.float32).unsqueeze(0) # (1, p, p)

        self.zero_mask = (torch.sum(self.N, dim=0) == 0).to(torch.float32)
        self.no_proliferate = self.D.unsqueeze(0).to(torch.float32).to(self.config['gpu_id'])

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

        for idx_time in range(self.N.shape[0]):
            K1_t, K2_t = function.get_K1_K2(self.N[idx_time])
            res_K1_t.append(K1_t)
            res_K2_t.append(K2_t)

        return torch.stack(res_K1_t), torch.stack(res_K2_t)
    
    def get_matrix_K(self, K_type='const', eval=False, tpoint=1.0):
        if K_type == 'const':
            K1 = torch.square(self.model.K1 * self.model.supplement[0] + self.model.supplement[1]) * self.model.K1_mask
            K2 = self.model.K2.squeeze() * self.model.supplement[2] + self.model.supplement[3]
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
                K1_t, K2_t = self.get_Kt(tpoint, self.model)
                matrix_K = self.combine_K1_K2(K1_t, K2_t)
                return matrix_K / self.exponent
            
            else:
                res_K1_t, res_K2_t = self.get_Kt_train(self.model) # (t, c, p, p) and (t, c, p)
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
        pena_nonL, pena_ub, pena_pop_zero, pena_k2_proli = \
            self.get_matrix_K(K_type=self.config['arch']['args']['K_type'])
        
        # thres = -1 if str(self.model_id).startswith('C') else int(self.config.num_epochs / 2)
        thres = 1500

        if epoch == thres:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.std.requires_grad = True

        if epoch > thres:
            std_reshape = torch.broadcast_to(self.model.std, self.N.shape)

            l_time3 = GaussianNLL(self.N[1] * self.sample_N[1], y_pred[1] * self.sample_N[1], torch.square(std_reshape[1] * 5.354))
            l_time10 = GaussianNLL(self.N[2] * self.sample_N[2], y_pred[2] * self.sample_N[2], torch.square(std_reshape[2] * 583.204))
            l_time17 = GaussianNLL(self.N[3] * self.sample_N[3], y_pred[3] * self.sample_N[3], torch.square(std_reshape[3] * 635.470))

            loss_obs = l_time3 + l_time10 + l_time17
            
        else:
            l_time3 = PoissonNLL(y_pred[1] * self.sample_N[1] / 5.354, self.N[1] * self.sample_N[1] / 5.354)
            l_time10 = PoissonNLL(y_pred[2] * self.sample_N[2] / 583.204, self.N[2] * self.sample_N[2] / 583.204)
            l_time17 = PoissonNLL(y_pred[3] * self.sample_N[3] / 635.470, self.N[3] * self.sample_N[3] / 635.470)
            
            loss_obs = l_time3 + l_time10 + l_time17

        l1 = self.config.alpha * torch.sum(pena_nonL)   
        l2 = self.config.alpha * torch.sum(pena_ub)
        l3 = 0.05 * torch.linalg.norm(pena_k2_proli, ord=1)
        l4 = 0.01 * torch.linalg.norm(pena_pop_zero, ord=2)

        if epoch > thres:
            l5 = torch.flatten((torch.square(self.model.std) > self.var_ub).to(torch.float32) * (torch.square(self.model.std) - self.var_ub))
            l5 = 0.01 * torch.linalg.norm(l5, ord=1)
        else:
            l5 = torch.tensor([0.0]).to(self.config.gpu)

        tb_scalar(
            ['Model/LR', 'Model/NFEForward'],
            [self.optimizer.param_groups[0]['lr'], self.model.nfe],
            epoch, self.writer
        )

        descrip = pbar_tb_description(
            ['ID', 'L/K2Pro', 'L/Upper', 'L/VarUb', 'L/Pop0', 'L/L3', 'L/L10', 'L/L17'],
            [
                self.model_id, 
                torch.max(pena_k2_proli).item(), 
                l2.item() / self.config.alpha, 
                l5.item() / 0.01, 
                torch.max(pena_pop_zero).item(),
                (SmoothL1(y_pred[1], self.N[1]) / 5).item(),
                (SmoothL1(y_pred[2], self.N[2]) / 583).item(),
                (SmoothL1(y_pred[3], self.N[3]) / 635).item()
            ],
            epoch, self.writer
        )
        pbar.set_description(descrip)
        self.model.nfe = 0

        if epoch > thres:
            loss = loss_obs + l5
        else:
            loss = loss_obs + l1 + l2 + l4 + l3 + l5
        loss.backward()
        return loss

    def train_model(self):
        '''
        For most problems, good choices are the default dopri5, 
        or to use rk4 with options=dict(step_size=...) set appropriately small. 
        Adjusting the tolerances (adaptive solvers) or step size (fixed solvers), 
        will allow for trade-offs between speed and accuracy.
        '''

        self.model.train()

        pbar = tqdm(range(self.config['base_trainer']['epochs']))
        for epoch in pbar:
            self.optimizer.zero_grad()

            if self.config['user_trainer']['adjoint']:
                y_pred = odeint_adjoint(self.model, self.N[0], self.t_observed, 
                    method='dopri5', rtol=1e-4, atol=1e-4, options=dict(dtype=torch.float32))
            else:
                y_pred = odeint(self.model, self.N[0], self.t_observed, 
                    method='dopri5', rtol=1e-4, atol=1e-4, options=dict(dtype=torch.float32))

            loss = self.compute_loss(y_pred, epoch, pbar)
            
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def eval_model(self, t_eval):
        self.model.eval()

        if self.config.adjoint:
            y_pred = odeint_adjoint(self.model, self.N[0], t_eval, method='dopri5')
        else:
            y_pred = odeint(self.model, self.N[0], t_eval, method='dopri5')
        
        return y_pred
