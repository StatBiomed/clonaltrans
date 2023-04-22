from torch import nn
import torch
import math
from torch.nn.parameter import Parameter

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
        act = nn.Softplus()
    if activation is None:
        def act(x):
            return x
    return act

class ODEBlock(nn.Module):
    def __init__(
        self, 
        N, 
        L, 
        hidden_dim: int = 16, 
        activation: str = 'softplus', 
        config: any = None
    ):
        super(ODEBlock, self).__init__()
        self.N = N
        self.L = torch.broadcast_to(L.unsqueeze(0), (N.shape[1], N.shape[2], N.shape[2]))
        self.nfe = 0
        self.config = config

        if config.num_layers == 1:
            #* ODE function dydt = K1 * y + K2 * y + bias (optional) with KaiMing Initialization
            self.K1, self.K2, self.offset = [], [], []

            for clone in range(N.shape[1]):
                K1 = torch.empty((N.shape[2], N.shape[2]))
                offset = torch.empty((N.shape[2], 1))
                K2 = torch.empty((1, N.shape[2]))
                self.reset_parameters(K1, offset)
                self.reset_parameters(K2, None)

                self.K1.append(K1.T)
                self.offset.append(offset.T)
                self.K2.append(K2.T)

            self.K1 = Parameter(torch.stack(self.K1), requires_grad=True)
            self.offset = Parameter(torch.stack(self.offset).squeeze(), requires_grad=True)
            self.K2 = Parameter(torch.stack(self.K2).squeeze(), requires_grad=True)

        if config.num_layers == 2:
            #* Batch processing 2 layer MLP for each individual clone with KaiMing Initialization
            self.encode, self.encode_bias, self.decode, self.decode_bias = [], [], [], []
            self.activation = activation_helper(activation)

            for clone in range(N.shape[1]):
                encode = torch.empty((hidden_dim, N.shape[2]))
                encode_bias = torch.empty((hidden_dim, 1))
                self.reset_parameters(encode, encode_bias)

                decode = torch.empty((N.shape[2], hidden_dim))
                decode_bias = torch.empty((N.shape[2], 1))
                self.reset_parameters(decode, decode_bias)

                self.encode.append(encode.T)
                self.encode_bias.append(encode_bias.T)
                self.decode.append(decode.T)
                self.decode_bias.append(decode_bias.T)
            
            self.encode = Parameter(torch.stack(self.encode), requires_grad=True)
            self.encode_bias = Parameter(torch.stack(self.encode_bias), requires_grad=True)
            self.decode = Parameter(torch.stack(self.decode), requires_grad=True)
            self.decode_bias = Parameter(torch.stack(self.decode_bias).squeeze(), requires_grad=True)

    def reset_parameters(self, weight, bias) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        if bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias, -bound, bound)

    def extra_repr(self) -> str:
        if self.config.num_layers == 1:
            return 'K1 (clone, pop, pop) = {}, \nK2 (clone, pop) = {}, \noffset = {}'.format(
                self.K1.shape, self.K2.shape, self.offset is not None
            )
        
        if self.config.num_layers == 2:
            return 'encode (clone, pop, hidden) = {}, \nencode_bias={}, \ndecode (clone, hidden, pop) = {}, \ndecode_bias={}'.format(
                self.encode.shape, self.encode_bias is not None, self.decode.shape, self.decode_bias is not None
            )

    def forward(self, t, y):
        self.nfe += 1

        if self.config.num_layers == 1:
            z = torch.bmm(y.unsqueeze(1), torch.square(self.K1) * self.L).squeeze()
            z = z + self.offset
            z = z + y * self.K2    
            return z
        
        if self.config.num_layers == 2:
            z = torch.bmm(y.unsqueeze(1), self.encode)
            z = z + self.encode_bias
            z = self.activation(z)

            z = torch.bmm(z, self.decode).squeeze()
            z = z + self.decode_bias
            return z

    def forward_direct(self, t, y):
        # Calculate K_influx and K_outflux matrices (m, p, p)
        # K_influx should be the transpose of K_outflux in dim 1 and 2
        K_base = self.K[-1].unsqueeze(0)
        K_outflux = torch.cat([self.K[:-1] + K_base, K_base], dim=0)
        # Calculate dydt for each clone in range [0, n-1]
        y = y.unsqueeze(1)
        dydt = torch.bmm(y, K_outflux.transpose(1, 2)) - torch.bmm(y, K_outflux)

        return dydt.squeeze() # (m, p)