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
        rtol=1e-5, options=dict(dtype=torch.float32),
    )

    loss = timeit(self.compute_loss, epoch, self.writer)(y_pred, epoch, pbar)