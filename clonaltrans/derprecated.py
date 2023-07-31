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

    def get_matrix_K(self):
        #* matrix_K (num_clones, num_populations, num_populations)
        #* matrix_K[-1] = base K(1) for background cells
        #* matrix_K[:-1] = parameter delta for each meta-clone specified in paper
        matrix_K = []
        for i in range(self.N.shape[1] - 1):
            matrix_K.append(
                torch.matmul(
                    self.ode_func.encode[i].weight.T, 
                    self.ode_func.decode[i].weight.T
                ) * self.L)
        
        matrix_K.append(
            torch.matmul(
                self.ode_func.encode[-1].weight.T, 
                self.ode_func.decode[-1].weight.T
            ) * self.L)
        return torch.stack(matrix_K)

class ODEBlock(torch.nn.Module):
    def __init__(self, N, hidden_dim) -> None:
        super().__init__()

        #* dydt = K1 * y + K2 * y + bias (optional)
        self.K1 = Parameter(torch.randn((N.shape[1], N.shape[2], N.shape[2])), requires_grad=True)
        self.K2 = Parameter(torch.randn((N.shape[1], N.shape[2])), requires_grad=True)
        self.offset = Parameter(torch.zeros((N.shape[1], N.shape[2])), requires_grad=True)
    
        #* Intuitely 2 layer MLP for each individual clone
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()

        for i in range(N.shape[1]):
            self.encode.append(nn.Linear(N.shape[2], hidden_dim))
            self.decode.append(nn.Linear(hidden_dim, N.shape[2]))

        #* Extracting weights and bias should be the same, whilst performance varies
        self.encode = nn.parameter.Parameter(torch.randn((N.shape[1], N.shape[2], hidden_dim)), requires_grad=True)
        self.encode_bias =  nn.parameter.Parameter(torch.zeros((N.shape[1], 1, hidden_dim)), requires_grad=True)
        self.decode = nn.parameter.Parameter(torch.randn((N.shape[1], hidden_dim, N.shape[2])), requires_grad=True)
        self.decode_bias =  nn.parameter.Parameter(torch.zeros((N.shape[1], N.shape[2])), requires_grad=True)
    
        # self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        # self.bias = Parameter(torch.empty(out_features, **factory_kwargs))

        # for p in self.ode_func.K1.parameters():
        #     p.data.clamp_(0)

    def forward(self, t, y):
        outputs = []

        for i in range(self.N.shape[1]):
            z = self.encode[i](y[i])
            z = self.activation(z)
            z = self.decode[i](z)
            outputs.append(z)

        return torch.stack(outputs)

def eval_predictions(model, t_observed, log_output=False, save=False):
    from .pl import mse_corr
    t_observed_norm = (t_observed - t_observed[0]) / (t_observed[-1] - t_observed[0])

    observations = model.input_N
    predictions = model.eval_model(t_observed_norm, log_output=log_output)

    mse_corr(observations[1:], predictions[1:], t_observed[1:].cpu().numpy(), save=save)

def get_matrix_K_log(self, eval=False):
    if self.config.num_layers == 1:
        K1_square = torch.square(self.ode_func.K1) # c*p*p
        matrix_K = self.input_N.unsqueeze(2) * K1_square.unsqueeze(0)
        
        if eval:
            K2_ = self.input_N * self.ode_func.K2.unsqueeze(0)
            for tp in range(matrix_K.shape[0]):
                for clone in range(matrix_K.shape[1]):
                    for pop in range(matrix_K.shape[2]):
                        matrix_K[tp, clone, pop, pop] += K2_[tp, clone, pop]
            
            return matrix_K
        else:
            return matrix_K * self.oppo_L_nondia, \
                torch.tensor([0.]), \
                torch.tensor([0.])
    
    if self.config.num_layers == 2:
        raise ValueError('2-layer model is currently not supported for time-variant K')

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

def simulation_2lK(
    encode,
    decode,
    config,
    t_simu=torch.tensor([0.0, 1.0, 2.0, 3.0]), 
    y0=None,
    noise_level=1e-2
):
    from .ode_block import ODEBlock
    ode_func = ODEBlock(
        num_clones=encode.shape[0],
        num_pops=encode.shape[1],
        hidden_dim=config.hidden_dim, 
        activation=config.activation, 
        num_layers=config.num_layers
    )

    ode_func.encode = Parameter(encode + torch.normal(0, 0.1, size=encode.shape).to(config.gpu), requires_grad=True)
    ode_func.decode = Parameter(decode + torch.normal(0, 0.1, size=decode.shape).to(config.gpu), requires_grad=True)

    array_total = odeint(ode_func, y0, t_simu, rtol=1e-5, method='dopri5', options=dict(dtype=torch.float32))

    scale_factor = array_total.abs()
    scale_factor[torch.rand_like(scale_factor) < 0.5] *= -1
    array_total = array_total + scale_factor * torch.rand_like(array_total) * noise_level

    array_total[array_total < 1] = 0
    array_total = torch.round(array_total)

    return array_total.to(config.gpu)

def get_hessian(model, t_observed):
    total_params = torch.concat([model.ode_func.K1.flatten(), model.ode_func.K2.flatten(), model.ode_func.std.flatten()])
    hessian = torch.zeros((total_params.shape[0], total_params.shape[0]))

    y_pred = odeint(model.ode_func, model.input_N[0], t_observed, method='dopri5', rtol=1e-5)
    loss = model.compute_loss(y_pred, None, None, hessian=True)
    loss.backward(create_graph=True)

    grad = model.ode_func.std.grad
    grad_flat = grad.view(-1)  # Flatten the gradient tensor
    print (grad_flat)
    
    hessian_size = grad_flat.shape[0]
    hessian_matrix = torch.zeros(hessian_size, hessian_size)

    for i, g in enumerate(grad_flat):
        if g.grad_fn is not None:  # Check if the gradient is not None
            print (i, g, g.grad_fn)
            g.backward(torch.ones_like(g), retain_graph=True)  # Compute the second-order gradient
            hessian_matrix[i] = model.ode_func.std.grad.view(-1)  # Store the second-order gradient in the Hessian matrix
            model.ode_func.std.grad.zero_()  # Clear the gradient for the next iteration

    # for i, param_i in enumerate(model.ode_func.K1.flatten()):
    #     for j, param_j in enumerate(model.ode_func.K1.flatten()):
    #         param_i.requires_grad_()
    #         grads1 = torch.autograd.grad(loss, param_i, create_graph=True)[0]
    #         grads2 = torch.autograd.grad(grads1, param_j, create_graph=True)[0]
            # hessian[i, j] = torch.autograd.grad(loss, param_j)[0]

    import torch
    import functorch
    from torch.nn.utils import _stateless

    model = torch.nn.Linear(2, 2)
    inp = torch.rand(1, 2)
    criterion = torch.nn.CrossEntropyLoss()

    def loss(params):
        out: torch.Tensor = _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, inp)
        return criterion (out, torch.zeros(len(inp), dtype=torch.long))

    names = list(n for n, _ in model.named_parameters())
    print(functorch.hessian(loss)(tuple(model.parameters())))

    return hessian

def streamline_per_epoch(self, data_dir, t_observed, epoch, loss):
    from .utils import TempModel
    temp = TempModel(data_dir=data_dir)
    t_pred = torch.linspace(t_observed[0], t_observed[-1], 100).to(self.config.gpu)
    predictions = self.eval_model(t_pred)

    grid_visual_interpolate(
        temp,
        [self.N, torch.pow(predictions, 1 / self.exponent), None],
        ['Observations', 'Predictions', None],
        [t_observed, t_pred, None],
        variance=False, 
        save=False
    )
    plt.title(f'Epoch {epoch}, Loss {loss:.3f}')

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