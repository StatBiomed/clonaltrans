import torch
from torch import nn
from tqdm import tqdm
from itertools import product
from utils import pbar_tb_description, time_func

GaussianNLL = nn.GaussianNLLLoss(reduction='mean', eps=1e-12)
PoissonNLL = nn.PoissonNLLLoss(log_input=False, reduction='mean')
SmoothL1 = nn.SmoothL1Loss()

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
        t_observed
    ):
        super(CloneTranModel, self).__init__()
        self.N = N
        self.L = L
        self.D = torch.tensor(config['user_trainer']['no_proliferation_pops'])
        self.model = model

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.t_observed = t_observed
        self.logger = config['data_loader']['args']['logger']

        self.config = config
        self.writer = writer
        self.scaling_factor = config['user_trainer']['scaling_factor']
        self.K_type = config['arch']['args']['K_type']
        self.gpu_id = self.config['system']['gpu_id']
        
        self.get_masks()

        self.var_ub = torch.square(torch.maximum(
            torch.max(self.N, 0)[0].unsqueeze(0) / 10, 
            torch.tensor([1e-6]).to(self.gpu_id)
        ))

    def get_masks(self):
        self.used_L = self.L.clone()
        self.used_L.fill_diagonal_(1)
        self.used_L = self.used_L.unsqueeze(0)

        self.oppo_L = (self.used_L == 0).to(torch.float32) # (c, p, p)
        self.zero_mask = (torch.sum(self.N, dim=0) == 0).to(torch.float32)
        self.no_proliferate = self.D.unsqueeze(0).to(torch.float32).to(self.gpu_id)

    def combine_K1_K2(self, K1, K2):
        for clone, pop in product(range(K1.shape[0]), range(K1.shape[1])):
            K1[clone, pop, pop] += K2[clone, pop]
        return K1

    def get_Kt(self, tpoint, function):
        t_evaluation = torch.tensor([0.0, max(tpoint, 0.001)]).to(self.gpu_id)
        predictions = self.eval_model(t_evaluation)[1]
        return function.get_K1_K2(predictions)
    
    def get_Kt_train(self, function):
        Kt_values = [function.get_K1_K2(self.N[idx]) for idx in range(self.N.shape[0])]
        return torch.stack([Kt[0] for Kt in Kt_values]), torch.stack([Kt[1] for Kt in Kt_values])

    def get_matrix_K(self, K_type='const', eval=False, tpoint=1.0):
        if K_type == 'const':
            return self._get_const_matrix_K(eval)

        elif K_type == 'dynamic':
            return self._get_dynamic_matrix_K(eval, tpoint)

    def _get_const_matrix_K(self, eval):
        self.K1 = torch.square(self.model.block.K1) * self.model.block.K1_mask # (c, p, p)
        self.K2 = self.model.block.K2.squeeze() # (c, p)

        if eval:
            return self.combine_K1_K2(self.K1, self.K2)
        else:
            return self._get_const_penalty(self.K1, self.K2)

    def _get_const_penalty(self, K1, K2):
        return self._get_flatten_values(K1, K2), \
            self._get_zero_mask_values(K1, K2), \
            self._get_no_proliferate_values(K2)

    def _get_dynamic_matrix_K(self, eval, tpoint):
        if eval:
            self.K1, self.K2 = self.get_Kt(tpoint, self.model.block) # (c, p, p) and (c, p)
            return self.combine_K1_K2(self.K1, self.K2)
        else:
            self.K1, self.K2 = self.get_Kt_train(self.model.block) # (t, c, p, p) and (t, c, p)
            return self._get_dynamic_penalty(self.K1, self.K2)

    def _get_dynamic_penalty(self, res_K1_t, res_K2_t):
        return self._get_flatten_values(res_K1_t, res_K2_t), \
            torch.cat([
                torch.flatten(res_K1_t * (self.N == 0).unsqueeze(-1).to(torch.float32)), 
                torch.flatten(res_K2_t * (self.N == 0).to(torch.float32))
            ]), \
            self._get_no_proliferate_values(torch.sum(res_K2_t, dim=0))

    def _get_flatten_values(self, K1, K2):
        return torch.cat([
            torch.flatten((K1 > self.config['user_trainer']['ub_for_diff']).to(torch.float32) * (K1 - self.config['user_trainer']['ub_for_diff'])), 
            torch.flatten((K2 > self.config['user_trainer']['ub_for_prol']).to(torch.float32) * (K2 - self.config['user_trainer']['ub_for_prol']))
        ])

    def _get_zero_mask_values(self, K1, K2):
        return torch.cat([torch.flatten(K1 * self.zero_mask.unsqueeze(-1)), torch.flatten(K2 * self.zero_mask)])

    def _get_no_proliferate_values(self, K2):
        return torch.flatten((K2 * self.no_proliferate > 0).to(torch.float32) * (K2 * self.no_proliferate))

    def freeze_model_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def compute_obs_loss(self, epoch, thres):
        loss_obs = 0.0

        for i in range(1, self.N.shape[0]):
            if epoch > thres:
                std_reshape = torch.broadcast_to(self.model.block.std, self.N.shape)
                loss_obs += GaussianNLL(self.N[i], self.y_pred[i], torch.square(std_reshape[i] * self.scaling_factor[i]))

            else:
                loss_obs += PoissonNLL(self.y_pred[i] / self.scaling_factor[i], self.N[i] / self.scaling_factor[i])

        return loss_obs

    def compute_penalty_terms(
        self, 
        epoch, 
        thres,
        pena_ub, 
        pena_pop_zero, 
        pena_k2_proli, 
        l4=torch.tensor(0.0, dtype=torch.float32)
    ):
        l1 = self.config['user_trainer']['alpha_1'] * torch.sum(pena_ub)
        l2 = self.config['user_trainer']['alpha_2'] * torch.linalg.norm(pena_k2_proli, ord=1)
        l3 = self.config['user_trainer']['alpha_3'] * torch.linalg.norm(pena_pop_zero, ord=2)

        if epoch > thres:
            l4 = torch.flatten((
                torch.square(self.model.block.std) > self.var_ub).to(torch.float32) 
                * (torch.square(self.model.block.std) - self.var_ub)
            )
            l4 = self.config['user_trainer']['alpha_4'] * torch.linalg.norm(l4, ord=1)
        
        K_total = []
        for idx, time in enumerate(self.t_observed):
            masks = torch.where(self.y_pred[idx] < torch.tensor([0.5]).to(self.gpu_id))

            if self.config['arch']['args']['K_type'] == 'const':
                self.K1 = self.K1.unsqueeze(0)
                self.K2 = self.K2.unsqueeze(0)
                idx = 0

            K = self.combine_K1_K2(self.K1[idx], self.K2[idx])

            if self.config['arch']['args']['K_type'] == 'const':
                self.K1 = self.K1.squeeze()
                self.K2 = self.K2.squeeze()

            K[masks[0], masks[1], :] = 0
            K_total.append(K)
        
        K_total = torch.stack(K_total)
        l5 = torch.mean(K_total[:, :-1, :, :], dim=1).flatten() - K_total[:, -1, :, :].flatten()
        l5 = self.config['user_trainer']['alpha_5'] * torch.linalg.norm(l5, ord=1)

        l6 = torch.flatten((K_total < 0).to(torch.float32) * K_total)
        l6 = self.config['user_trainer']['alpha_6'] * torch.linalg.norm(torch.abs(l6), ord=2)

        return l1, l2, l3, l4, l5, l6

    def compute_loss(self, epoch, pbar):
        pena_ub, pena_pop_zero, pena_k2_proli = self.get_matrix_K(K_type=self.K_type)
        thres = self.config['optimizer']['thres']

        if epoch == thres:
            self.freeze_model_parameters()
            self.model.block.std.requires_grad = True

        loss_obs = self.compute_obs_loss(epoch, thres)
        l1, l2, l3, l4, l5, l6 = self.compute_penalty_terms(
            epoch, 
            thres,
            pena_ub, 
            pena_pop_zero, 
            pena_k2_proli
        )

        descrip = pbar_tb_description(
            ['L/Upper', 'L/K2Pro', 'L/Pop0', 'L/VarUb', 'L/DiffBG', 'L/Neg0', 'L/L3', 'L/L10', 'L/L17'],
            [ 
                l1.item() / self.config['user_trainer']['alpha_1'], 
                l2.item() / self.config['user_trainer']['alpha_2'],
                l3.item() / self.config['user_trainer']['alpha_3'],
                l4.item() / self.config['user_trainer']['alpha_4'],
                l5.item() / self.config['user_trainer']['alpha_5'],
                l6.item() / self.config['user_trainer']['alpha_6'],
                (SmoothL1(self.y_pred[1], self.N[1]) / self.scaling_factor[1]).item(),
                (SmoothL1(self.y_pred[2], self.N[2]) / self.scaling_factor[2]).item(),
                (SmoothL1(self.y_pred[3], self.N[3]) / self.scaling_factor[3]).item()
            ],
            epoch, self.writer
        )
        pbar.set_description(descrip)

        loss = loss_obs + l4 if epoch > thres else loss_obs + l1 + l2 + l3 + l4 + l5 + l6
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
            
            self.y_pred = self.model(self.N[0], self.t_observed)
            loss = self.compute_loss(epoch, pbar)
            
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def eval_model(self, t_eval):
        self.model.eval()
        return self.model(self.N[0], t_eval)
