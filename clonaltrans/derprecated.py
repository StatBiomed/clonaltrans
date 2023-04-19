import time
import inspect
import torch

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

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

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

class Deprecated(torch.nn.Module):
    def __init__(self) -> None:

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