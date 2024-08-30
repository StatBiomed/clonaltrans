from torch import nn
import torch
import math
from torch.nn.parameter import Parameter
from torchdiffeq import odeint_adjoint, odeint
from utils import time_func

def activation_helper(activation):
    if activation == 'gelu':
        act = nn.GELU()
    if activation == 'sigmoid':
        act = nn.Sigmoid()
    if activation == 'tanh':
        act = nn.Tanh()
    if activation == 'elu':
        act = nn.ELU(alpha=1.0)
    if activation == 'relu':
        act = nn.ReLU()
    if activation == 'softplus':
        act = nn.Softplus(threshold=7)
    if activation is None:
        def act(x):
            return x
    return act

class ODEBlock(nn.Module):
    def __init__(
        self, 
        L,
        num_clones: int = 7,
        num_pops: int = 11,
        hidden_dim: int = 32, 
        activation: str = 'softplus', 
        K_type: str = 'const'
    ):
        '''
        ODE function dydt = K1 * y + K2 * y + K1.T * y 
        K1, >= 0, upper off-diagonal, transitions between populations (both inbound & outbound), constraint by PAGA L
        K2, diagonal, can be negative, ideally maximum 6.0, self proliferation and cell apoptosis process
        ''' 
        super(ODEBlock, self).__init__()

        self.K_type = K_type
        self.activation = activation_helper(activation)
        self.K1_mask = Parameter(L.unsqueeze(0), requires_grad=False)

        if self.K_type == 'const':
            self.K1, self.K2 = self.get_const(num_clones, num_pops)
        
        if self.K_type == 'dynamic':
            self.K1_encode, self.K1_decode, self.K2_encode, self.K2_decode = self.get_dynamic(num_clones, num_pops, hidden_dim)

        self.layernorm = nn.LayerNorm(num_pops)
        self.forward_func = self._const_forward if self.K_type == 'const' else self._dynamic_forward

    def get_const(self, num_clones, num_pops):
        K1, _ = self.linear_init(num_clones, num_pops, num_pops, 'kaiming_uniform')
        K2, _ = self.linear_init(num_clones, num_pops, 1, 'kaiming_uniform')
        return K1, K2

    def get_dynamic(self, num_clones, num_pops, hidden_dim):
        K1_encode, _ = self.linear_init(num_clones, num_pops, hidden_dim, 'normal')
        K1_decode, _ = self.linear_init(num_clones, hidden_dim, num_pops * num_pops, 'normal')
        K2_encode, _ = self.linear_init(num_clones, num_pops, hidden_dim, 'normal')
        K2_decode, _ = self.linear_init(num_clones, hidden_dim, num_pops, 'normal')
        return K1_encode, K1_decode, K2_encode, K2_decode

    def reset_parameters(self, weight, bias, basis='normal') -> None:
        if basis == 'kaiming_uniform':
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if basis == 'normal':
            nn.init.normal_(weight, 0, 0.01)

        if bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias, -bound, bound)

    def linear_init(self, num_clones, in_dim, out_dim, basis):
        weight_matrix, bias_matrix = [], []

        for clone in range(num_clones):
            weight = torch.empty((out_dim, in_dim))
            bias = torch.empty((out_dim, ))

            self.reset_parameters(weight, bias, basis)

            weight_matrix.append(weight.T)
            bias_matrix.append(bias)

        return Parameter(torch.stack(weight_matrix) / 2, requires_grad=True), Parameter(torch.stack(bias_matrix) / 2, requires_grad=True)

    def get_K1_K2(self, y):
        y = self.layernorm(y)

        z1 = torch.bmm(y.unsqueeze(1), self.K1_encode)
        z1 = self.activation(z1)
        z1 = torch.bmm(z1, self.K1_decode)
        z1 = torch.square(z1.squeeze().view(y.shape[0], y.shape[1], y.shape[1])) * self.K1_mask

        z2 = torch.bmm(y.unsqueeze(1), self.K2_encode)
        z2 = self.activation(z2)
        z2 = torch.bmm(z2, self.K2_decode)
        return z1, z2.squeeze()

    def _const_forward(self, t, y):
        z = torch.bmm(y.unsqueeze(1), torch.square(self.K1) * self.K1_mask).squeeze()
        z += y * self.K2.squeeze()
        z -= torch.sum(y.unsqueeze(1) * torch.square(torch.transpose(self.K1, 1, 2)) * torch.transpose(self.K1_mask, 1, 2), dim=1)
        return z

    def _dynamic_forward(self, t, y):
        K1_t, K2_t = self.get_K1_K2(y)
        z = torch.bmm(y.unsqueeze(1), K1_t).squeeze()
        z += y * K2_t 
        z -= torch.sum(y.unsqueeze(1) * torch.transpose(K1_t, 1, 2), dim=1)
        return z

    def forward(self, t, y):
        return self.forward_func(t, y)

    def extra_repr(self) -> str:
        if self.K_type == 'const':
            return 'K1 (clone, pop, pop) = {}, \nK2 (clone, pop) = {}'.format(
                self.K1.shape, self.K2.squeeze().shape
            )

        if self.K_type == 'dynamic':
            expr1 = 'K1_encode (clone, pop, hidden) = {}, \nK1_decode (clone, hidden, pop * pop) = {}'.format(
                self.K1_encode.shape, self.K1_decode.shape
            )
            expr2 = 'K2_encode (clone, pop, hidden) = {}, \nK2_decode (clone, hidden, pop) = {}'.format(
                self.K2_encode.shape, self.K2_decode.shape
            )
            return expr1 + '\n' + expr2

class ODESolver(nn.Module):
    def __init__(
        self, 
        L,
        num_clones: int = 7,
        num_pops: int = 11,
        hidden_dim: int = 32, 
        activation: str = 'softplus', 
        K_type: str = 'const',
        adjoint: bool = False,
    ):
        super(ODESolver, self).__init__()

        self.block = ODEBlock(L, num_clones, num_pops, hidden_dim, activation, K_type)
        self.adjoint = adjoint
    
    def forward(
        self, 
        N0, 
        t_observed, 
        methods='dopri5',
        rtol=1e-4,
        atol=1e-4,
        options=dict(dtype=torch.float32)
    ):
        if self.adjoint:
            return odeint_adjoint(self.block, N0, t_observed, method=methods, rtol=rtol, atol=atol, options=options)
        
        else:
            return odeint(self.block, N0, t_observed, method=methods, rtol=rtol, atol=atol, options=options)