import time
import inspect
import torch
from torch import nn
from torch.nn.parameter import Parameter

def timeit(section=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.monotonic()
            result = func(*args, **kwargs)
            end_time = time.monotonic()
            # Use inspect to get the line number of the decorated function call
            line_number = inspect.currentframe().f_back.f_lineno
            if section is None or section.lower() == 'all':
                print(f"Line {line_number} in {func.__name__} took {end_time - start_time:.6f} seconds")
            elif section.lower() == 'forward':
                print(f"Forward pass in {func.__name__} took {end_time - start_time:.6f} seconds")
            elif section.lower() == 'backward':
                print(f"Backward pass in {func.__name__} took {end_time - start_time:.6f} seconds")
            elif section.lower() == 'update':
                print(f"Parameter update in {func.__name__} took {end_time - start_time:.6f} seconds")
            return result
        return wrapper
    return decorator

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    @timeit(section='forward')
    def forward(self, x):
        x = self.linear(x)
        return x

    @timeit(section='backward')
    def backward(self, loss):
        loss.backward()

    @timeit(section='update')
    def update(self, optimizer):
        optimizer.step()

class Deprecated(nn.Module):
    def __init__(self, L, epoch, config) -> None:

        self.non_diagonal = torch.ones(L.shape).fill_diagonal_(0).unsqueeze(0).to(config.gpu)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.config.lrs_step, 
            gamma=self.config.lrs_gamma
        )

        self.writer.add_scalar('NFE/Backward', self.ode_func.nfe, epoch)
        self.ode_func.nfe = 0

class ODEBlock(torch.nn.Module):
    def __init__(self, N, hidden_dim) -> None:
        super().__init__()

        #* dydt = K1 * y + K2 * y + bias (optional)
        self.K1 = Parameter(torch.randn((N.shape[1], N.shape[2], N.shape[2])), requires_grad=True)
        self.K2 = Parameter(torch.randn((N.shape[1], N.shape[2])), requires_grad=True)
        self.offset = Parameter(torch.zeros((N.shape[1], N.shape[2])), requires_grad=True)

        for p in self.ode_func.K1.parameters():
            p.data.clamp_(0)

def train(self, epoch, t_observed, pbar):
    y_pred = timeit(
        odeint_adjoint if self.config.adjoint else odeint, epoch, self.writer
    )(
        self.ode_func, self.input_N[0], t_observed, method='dopri5',
        rtol=1e-4, atol=1e-4, options=dict(dtype=torch.float32), 
    )

    loss = timeit(self.compute_loss, epoch, self.writer)(y_pred, epoch, pbar)

def get_scheduler(
    optimizer, 
    num_warmup_steps, 
    d_model=1.0, 
    last_epoch=-1
):
    '''
    scheduler = get_scheduler(opt, 200, d, -1)
    '''
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        current_step += 1
        arg1 = current_step ** -0.5
        arg2 = current_step * (num_warmup_steps ** -1.5)
        return (d_model ** -0.5) * min(arg1, arg2)
    return LambdaLR(optimizer, lr_lambda, last_epoch)

class Mixture(nn.Module):
    def __init__(self) -> None:
        super(Mixture, self).__init__()
        print (f'{config.K_type}', '\n', self.ode_func_dyna) if config.K_type == 'mixture_lr' else print ('')

        if self.K_type == 'mixture':
            self.K1, self.K2 = self.get_const(num_clones, num_pops)
            self.K1_encode, self.K1_decode, self.K2_encode, self.K2_decode = self.get_dynamic(num_clones, num_pops, hidden_dim)

    def init_optimizer(self):
        if self.config.K_type == 'mixture_lr':
            self.optimizer_dyna = torch.optim.Adam(self.ode_func_dyna.parameters(), lr=0.001, amsgrad=True)
            self.scheduler_dyna = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_dyna, milestones=[100 * i for i in range(1, 10)], gamma=0.5)

    def get_matrix_K(self, K_type='const', eval=False, tpoint=1.0, sep='mixture'):
        if K_type == 'mixture':
            K1 = torch.square(self.ode_func.K1) * self.ode_func.K1_mask
            K2 = self.ode_func.K2.squeeze()

            if eval:
                function = self.ode_func_dyna if self.config.K_type == 'mixture_lr' else self.ode_func
                fraction = 2 if self.config.K_type == 'mixture_lr' else 1
                assert sep in ['const', 'dynamic', 'mixture']
                K1_t, K2_t = self.get_Kt(tpoint, function)

                if sep == 'const':
                    return self.combine_K1_K2(K1, K2) / self.exponent / fraction
                if sep == 'dynamic':
                    return self.combine_K1_K2(K1_t, K2_t) / self.exponent / fraction
                if sep == 'mixture':
                    return self.combine_K1_K2(K1 + K1_t, K2 + K2_t) / self.exponent / fraction
            
            else:
                function = self.ode_func_dyna if self.config.K_type == 'mixture_lr' else self.ode_func
                K1, K2 = K1.unsqueeze(0), K2.unsqueeze(0)
                res_K1_t, res_K2_t = self.get_Kt_train(function) # (t, c, p, p) and (t, c, p)

                return torch.concat([K1, res_K1_t]) / self.exponent * self.oppo_L_nondia.unsqueeze(0), \
                    ((torch.concat([K2, res_K2_t]) / self.exponent - 6) > 0).to(torch.float32) * (torch.concat([K2, res_K2_t]) / self.exponent - 6), \
                    torch.concat([
                        torch.flatten(torch.concat([K1, res_K1_t]) / self.exponent * (self.N == 0).unsqueeze(-1).to(torch.float32)), 
                        torch.flatten(torch.concat([K2, res_K2_t]) / self.exponent * (self.N == 0).to(torch.float32))
                    ]), \
                    torch.concat([torch.flatten(function.K1_decode), torch.flatten(function.K2_decode)])
        
    def train_model(self, t_observed):
        if self.config.K_type == 'mixture_lr':
            self.optimizer_dyna.zero_grad()

            y_pred_dyna = odeint(self.ode_func_dyna, self.input_N[0], t_observed, 
                method='dopri5', rtol=1e-4, atol=1e-4, options=dict(dtype=torch.float32))
            # y_pred_dyna = odeint(self.ode_func_dyna, self.input_N[0], t_observed, 
            #     method='rk4', options=dict(step_size=0.1))

            y_pred = (y_pred + y_pred_dyna) / 2

        if self.config.K_type == 'mixture_lr':
            self.optimizer_dyna.step()
            self.scheduler_dyna.step()
        
    def eval_model(self, t_eval):
        if self.config.K_type == 'mixture_lr':
            self.ode_func_dyna.eval()
            y_pred_dyna = odeint(self.ode_func_dyna, self.input_N[0], t_eval, method='dopri5')
            y_pred = (y_pred + y_pred_dyna) / 2
    
        elif self.config.input_form == 'log' and log_output:
            return y_pred
        
        else: return (torch.exp(y_pred) - 1.0)
    
    def forward(self, y):
        if self.K_type == 'mixture':
            z1 = torch.bmm(y.unsqueeze(1), torch.square(self.K1) * self.K1_mask).squeeze()
            z1 += y * self.K2.squeeze()
            z1 -= torch.sum(y.unsqueeze(1) * torch.square(self.K1.mT) * self.K1_mask.mT, dim=1)

            K1_t, K2_t = self.get_K1_K2(y)
            z2 = torch.bmm(y.unsqueeze(1), K1_t).squeeze()
            z2 += y * K2_t 
            z2 -= torch.sum(y.unsqueeze(1) * K1_t.mT, dim=1)

            # z = torch.sigmoid(self.lam) * z1 + (1 - torch.sigmoid(self.lam)) * z2
            # z = self.lam * z1 + (1 - self.lam) * z2
            z = z1 + z2
            return z
    
    def extra_repr(self) -> str:
        if self.K_type == 'mixture':
            expr1 = 'K1 (clone, pop, pop) = {}, \nK2 (clone, pop) = {}'.format(
                self.K1.shape, self.K2.squeeze().shape
            )
            expr2 = 'K1_encode (clone, pop, hidden) = {}, \nK1_decode (clone, hidden, pop * pop) = {}'.format(
                self.K1_encode.shape, self.K1_decode.shape
            )
            expr3 = 'K2_encode (clone, pop, hidden) = {}, \nK2_decode (clone, hidden, pop) = {}'.format(
                self.K2_encode.shape, self.K2_decode.shape
            )
            return expr1 + '\n' + expr2 + '\n' + expr3

def trajectory_range(model_list, model_ref, raw_data=True):
    model_list.append(model_ref)
    pbar = tqdm(enumerate(model_list))
    total_pred = []

    for idx, model in pbar:
        model.ode_func.supplement = [model.ode_func.supplement[i].to('cpu') for i in range(4)]
        t_smoothed = torch.linspace(model.t_observed[0], model.t_observed[-1], 100).to('cpu')
        y_pred = model.eval_model(t_smoothed)

        #TODO fit for different data transformation techniques
        if raw_data:
            data_values = [model.N, torch.pow(y_pred, 1 / model_ref.config.exponent), None]
        else:
            data_values = [model.input_N, y_pred, None]

        obs, pred, _ = data_convert(data_values)
        total_pred.append(pred)
    
    return np.stack(total_pred), obs, t_smoothed

def trajectory_ci(
    total_pred,
    obs,
    t_smoothed,
    model_ref,
    boundary,
    save: bool = False
):
    fig, axes = plt.subplots(model_ref.N.shape[1], model_ref.N.shape[2], figsize=(40, 15), sharex=True)
    lb, ub = np.percentile(total_pred, boundary[0], axis=0), np.percentile(total_pred, boundary[1], axis=0)

    t_obs, t_pred, t_median = data_convert([model_ref.t_observed, t_smoothed, t_smoothed])
    data_names = ['Observations', 'Predictions', 'Q50']
    anno = pd.read_csv(os.path.join(model_ref.data_dir, 'annotations.csv'))
    sample_N = np.ones(obs.shape)

    for row in range(model_ref.N.shape[1]):
        for col in range(model_ref.N.shape[2]):
            axes[row][col].fill_between(
                t_pred,
                lb[:, row, col],
                ub[:, row, col],
                color='lightskyblue',
                alpha=0.5
            )

            size_samples = sample_N[:, row, col]
            plot_gvi(np.percentile(total_pred, 50, axis=0), axes, row, col, t_median, data_names[2], '#929591', size_samples)
            plot_gvi(total_pred[-1], axes, row, col, t_pred, data_names[1], 'lightcoral', size_samples)
            plot_gvi(obs, axes, row, col, t_obs, data_names[0], '#2C6975', size_samples)
    
            axes[0][col].set_title(anno['populations'][col])
            axes[row][0].set_ylabel(anno['clones'][row])
            axes[row][col].set_xticks(t_obs, labels=t_obs.astype(int), rotation=45)
            axes[row][col].ticklabel_format(axis='y', style='sci', scilimits=(0, 4))

    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.5)

    legend_elements = [
        Line2D([0], [0], marker='o', color='#2C6975', markersize=7, linestyle=''), 
        Line2D([0], [0], color='lightcoral', lw=2), 
        Line2D([0], [0], color='lightskyblue', lw=2), 
        Line2D([0], [0], color='#929591', lw=2)
    ]
    labels = ['Observations', 'Predictions', f'Q{boundary[0]} - Q{boundary[1]}', 'Q50']
    fig.legend(legend_elements, labels, loc='right', fontsize='x-large', bbox_to_anchor=(0.96, 0.5))

    if save:
        plt.savefig(f'./figs/{save}.png', dpi=300, bbox_inches='tight')

def const_and_dyna(K_const, K_dyna, save=False):
    assert K_const.shape == K_dyna.shape
    from .pl import get_subplot_dimensions
    rows, cols, figsize = get_subplot_dimensions(K_const.shape[0], max_cols=3, fig_height_per_row=2)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for n in range(K_const.shape[0]):
        x = K_const[n].flatten()
        y = K_dyna[n].flatten()
        
        ax_loc = axes[n // cols][n % cols] if rows > 1 else axes[n]
        sns.scatterplot(x=x, y=y, s=25, ax=ax_loc, c='lightcoral')
        ax_loc.plot([x.min(), x.max()], [x.min(), x.max()], linestyle="--", color="grey")

        ax_loc.set_title(f'Clone {n}', fontsize=10)
        ax_loc.set_ylabel(f'K_dynamic')

        if n > 2:
            ax_loc.set_xlabel(f'K_const')

    if save is not False:
        plt.savefig(f'./figs/{save}.png', dpi=300, bbox_inches='tight')