import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from itertools import product
import sys
from utils import pbar_tb_description, time_func
import os

GaussianNLL = nn.GaussianNLLLoss(reduction='mean', eps=1e-12)
PoissonNLL = nn.PoissonNLLLoss(log_input=False, reduction='mean')
SmoothL1 = nn.SmoothL1Loss()

class CloneTranModel(nn.Module):
    def __init__(
        self,
        N, # (num_timpoints, num_clones, num_populations)
        L, # (num_populations, num_populations)
        config,
        model,
        optimizer,
        t_observed,
        trainer_type='training',
        writer=None,
        sample_N=None,
        gpu_id=None
    ):
        super(CloneTranModel, self).__init__()
        self.N = N
        self.L = L
        self.D = torch.tensor(config['user_trainer']['no_proliferation_pops'])
        self.A = torch.tensor(config['user_trainer']['no_apoptosis_pops'])
        self.model = model

        self.optimizer = optimizer
        self.t_observed = t_observed

        self.config = config
        self.scaling_factor = torch.tensor(config['user_trainer']['scaling_factor'], dtype=torch.float32).to(gpu_id)
        self.K_type = config['arch']['args']['K_type']
        self.gpu_id = gpu_id

        self.trainer_type = trainer_type
        self.writer = writer
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400, 600, 800, 1000], gamma=0.5)
        
        if self.trainer_type == 'training':
            self.logger = config['data_loader']['args']['logger']
            self.model_id = 0.0

        elif self.trainer_type == 'bootstrapping':
            self.sample_N = sample_N

        elif self.trainer_type == 'simulation':
            self.model_id = 0.0

        else:
            raise ValueError('Invalid trainer_type type')
        
        self.get_masks()
        self.get_weight_by_metaclone_size()

    def get_weight_by_metaclone_size(self):
        total_counts = torch.sum(self.N, dim=(0, 2))[:-1]
        self.rate_weights = total_counts / torch.sum(total_counts)

        if self.trainer_type == 'training':
            self.logger.info(f'Rate weights for each meta-clones: {self.rate_weights}')

    def get_masks(self):
        self.used_L = self.L.clone()
        self.used_L.fill_diagonal_(1)
        self.used_L = self.used_L.unsqueeze(0)

        self.oppo_L = (self.used_L == 0).to(torch.float32) # (c, p, p)
        self.zero_mask = (torch.sum(self.N, dim=0) == 0).to(torch.float32)
        
        self.no_proliferate = self.D.unsqueeze(0).to(torch.float32).to(self.gpu_id)
        self.no_apoptosis = self.A.unsqueeze(0).to(torch.float32).to(self.gpu_id)

    def combine_K1_K2(self, K1, K2):
        idx = torch.arange(K1.shape[1], device=K1.device)

        for clone in range(K1.shape[0]):
            K1[clone, idx, idx] += K2[clone]

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
            return self._get_const_matrix_K(eval, 'const')

        elif K_type == 'dynamic':
            return self._get_dynamic_matrix_K(eval, tpoint, 'dynamic')

    def _get_const_matrix_K(self, eval, K_type):
        self.K1 = torch.square(self.model.block.K1) * self.model.block.K1_mask # (c, p, p)
        self.K2 = self.model.block.K2.squeeze() # (c, p)

        if eval:
            return self.combine_K1_K2(self.K1, self.K2)
        else:
            return self._get_penalties(self.K1, self.K2, K_type)

    def _get_dynamic_matrix_K(self, eval, tpoint, K_type):
        if eval:
            self.K1, self.K2 = self.get_Kt(tpoint, self.model.block) # (c, p, p) and (c, p)
            return self.combine_K1_K2(self.K1, self.K2)
        else:
            self.K1, self.K2 = self.get_Kt_train(self.model.block) # (t, c, p, p) and (t, c, p)
            return self._get_penalties(self.K1, self.K2, K_type)

    def _get_penalties(self, K1, K2, K_type):
        return self._get_flatten_values(K1, K2), \
            self._get_zero_mask_values(K1, K2, K_type), \
            self._get_no_proliferate_values(K2), \
            self._get_no_apoptosis_values(K2)

    def _get_flatten_values(self, K1, K2):
        return torch.cat([
            torch.flatten((K1 > self.config['user_trainer']['ub_for_diff']).to(torch.float32) * (K1 - self.config['user_trainer']['ub_for_diff'])), 
            torch.flatten((K2 > self.config['user_trainer']['ub_for_prol']).to(torch.float32) * (K2 - self.config['user_trainer']['ub_for_prol']))
        ])

    def _get_zero_mask_values(self, K1, K2, K_type):
        if K_type == 'const':
            return torch.cat([torch.flatten(K1 * self.zero_mask.unsqueeze(-1)), torch.flatten(K2 * self.zero_mask)])
        
        if K_type == 'dynamic':
            return torch.cat([
                torch.flatten(K1 * (self.N == 0).unsqueeze(-1).to(torch.float32)), 
                torch.flatten(K2 * (self.N == 0).to(torch.float32))
            ])

    def _get_no_proliferate_values(self, K2):
        return torch.flatten((K2 * self.no_proliferate > 0).to(torch.float32) * (K2 * self.no_proliferate))

    def _get_no_apoptosis_values(self, K2):
        return torch.flatten((K2 * self.no_apoptosis < 0).to(torch.float32) * (K2 * self.no_apoptosis))

    def freeze_model_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def compute_obs_loss_train(self):
        loss_obs = 0.0
        for i in range(1, self.N.shape[0]):
            loss_obs += PoissonNLL(self.y_pred[i] / self.scaling_factor[i], self.N[i] / self.scaling_factor[i])
        return loss_obs

    def compute_obs_loss_boots(self):
        loss_obs = 0.0
        for i in range(1, self.N.shape[0]):
            loss_obs += PoissonNLL(self.y_pred[i] * self.sample_N[i] / self.scaling_factor[i], self.N[i] * self.sample_N[i] / self.scaling_factor[i])
        return loss_obs

    # def compute_obs_loss_train(self):
    #     scaling_factor = self.scaling_factor[1:].unsqueeze(-1).unsqueeze(-1)
    #     y_pred_scaled = self.y_pred[1:] / scaling_factor
    #     N_scaled = self.N[1:] / scaling_factor
    #     return PoissonNLL(y_pred_scaled, N_scaled)

    # def compute_obs_loss_boots(self):
    #     scaling_factor = self.scaling_factor[1:].unsqueeze(-1).unsqueeze(-1)
    #     y_pred_scaled = self.y_pred[1:] * self.sample_N[1:] / scaling_factor
    #     N_scaled = self.N[1:] * self.sample_N[1:] / scaling_factor
    #     return PoissonNLL(y_pred_scaled, N_scaled)

    def compute_penalty_l0_l1_l2(self, p_uppb, p_prol, p_zero):
        l0 = self.config['user_trainer']['alphas'][0] * torch.sum(p_uppb)
        l1 = self.config['user_trainer']['alphas'][1] * torch.linalg.norm(p_prol, ord=1)
        l2 = self.config['user_trainer']['alphas'][2] * torch.linalg.norm(p_zero, ord=2)
        return l0, l1, l2

    def compute_penalty_l3_l4_l5(self, K_total, p_apop):
        # l3 = K_total[:, :-1, :, :] * self.rate_weights.unsqueeze(-1).unsqueeze(-1)
        # l3 = torch.sum(l3, dim=1).flatten() - K_total[:, -1, :, :].flatten()
        # l3 = self.config['user_trainer']['alphas'][3] * torch.linalg.norm(torch.abs(l3), ord=1)

        l3 = torch.mean(K_total[:, :-1, :, :], dim=1).flatten() - K_total[:, -1, :, :].flatten()
        l3 = self.config['user_trainer']['alphas'][3] * torch.linalg.norm(torch.abs(l3), ord=1)

        l4 = torch.flatten((K_total < 0).to(torch.float32) * K_total) / 2
        l4 = self.config['user_trainer']['alphas'][4] * torch.linalg.norm(torch.abs(l4), ord=2)

        l5 = self.config['user_trainer']['alphas'][5] * torch.linalg.norm(p_apop, ord=1)
        return l3, l4, l5

    def compute_penalty_terms(
        self, 
        p_uppb, 
        p_zero, 
        p_prol,
        p_apop
    ):
        l0, l1, l2 = self.compute_penalty_l0_l1_l2(p_uppb, p_prol, p_zero)
        
        K_total = []
        for idx, time in enumerate(self.t_observed):        
            masks = torch.where(self.y_pred[idx] < torch.tensor([0.5]).to(self.gpu_id))
                
            input_K1 = self.K1 if self.config['arch']['args']['K_type'] == 'const' else self.K1[idx]
            input_K2 = self.K2 if self.config['arch']['args']['K_type'] == 'const' else self.K2[idx]

            K = self.combine_K1_K2(input_K1, input_K2)
            K[masks[0], masks[1], :] = 0
            K_total.append(K)
        
        K_total = torch.stack(K_total)
        
        l3, l4, l5 = self.compute_penalty_l3_l4_l5(K_total, p_apop)
        return l0, l1, l2, l3, l4, l5

    def compute_loss(self, epoch, pbar):
        p_uppb, p_zero, p_prol, p_apop = self.get_matrix_K(K_type=self.K_type)

        if False:
            self.freeze_model_parameters()

        if self.trainer_type == 'training':
            loss_obs = self.compute_obs_loss_train()
        elif self.trainer_type == 'bootstrapping':
            loss_obs = self.compute_obs_loss_boots()
        elif self.trainer_type == 'simulation':
            loss_obs = self.compute_obs_loss_train()
        else:
            raise ValueError('Invalid trainer type')

        l0, l1, l2, l3, l4, l5 = self.compute_penalty_terms(
            p_uppb, 
            p_zero, 
            p_prol,
            torch.abs(p_apop)
        )

        descrip = pbar_tb_description(
            ['ID', 'L/K2Pro', 'L/Pop0', 'L/DiffBG', 'L/Neg0', 'L/Apop', 'L/Recon'],
            [ 
                self.model_id, 
                l1.item() / self.config['user_trainer']['alphas'][1],
                l2.item() / self.config['user_trainer']['alphas'][2],
                l3.item() / self.config['user_trainer']['alphas'][3],
                l4.item() / self.config['user_trainer']['alphas'][4],
                l5.item() / self.config['user_trainer']['alphas'][5],
                np.sum([(SmoothL1(self.y_pred[i], self.N[i]) / self.scaling_factor[i]).item() for i in range(1, self.N.shape[0])]),
            ],
            epoch, self.writer
        )
        pbar.set_description(descrip)

        loss = loss_obs + l0 + l1 + l2 + l3 + l4 + l5
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

        lr = self.config['optimizer']['learning_rate']
        losses = [sys.maxsize]  
        min_loss = losses[-1]
        model_saved = False

        pbar = tqdm(range(self.config['base_trainer']['epochs']))
        for epoch in pbar:
            self.optimizer.zero_grad()
            
            self.y_pred = self.model(self.N[0], self.t_observed)
            loss = self.compute_loss(epoch, pbar)
            
            self.optimizer.step()
            # self.scheduler.step()
    
            if self.trainer_type == 'training':
                losses.append(loss.cpu().detach().numpy())
                self.writer.add_scalar("LearningRate", lr, epoch)
                
                if epoch % self.config['base_trainer']['log_interval'] == 0 and epoch > 0: 
                    if (not np.isnan(losses[-1])) and (losses[-1] < min_loss):
                        min_loss = losses[-1]
                        torch.save(self.model.state_dict(), os.path.join(self.config.save_dir, f'model_last.cpt'))
                        model_saved = True

                    else:
                        if model_saved:
                            self.model.load_state_dict(torch.load(os.path.join(self.config.save_dir, f'model_last.cpt')))
                            self.model = self.model.to(self.gpu_id)

                        lr *= self.config['optimizer']['lr_decay']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr

        if losses[-1] < min_loss:
            torch.save(self.model.state_dict(), os.path.join(self.config.save_dir, f'model_last.cpt'))

    @torch.no_grad()
    def eval_model(self, t_eval):
        self.model.eval()
        return self.model(self.N[0], t_eval)
