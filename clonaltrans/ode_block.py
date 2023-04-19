from torch import nn
import torch

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
        activation: str = 'gelu', 
        config: any = None
    ):
        super(ODEBlock, self).__init__()
        self.N = N
        self.L = torch.broadcast_to(L.unsqueeze(0), (N.shape[1], N.shape[2], N.shape[2]))
        self.nfe = 0
        self.config = config
        self.activation = activation_helper(activation)

        if config.num_layers == 1:
            self.K1 = nn.parameter.Parameter(torch.randn((N.shape[1], N.shape[2], N.shape[2])), requires_grad=True)
            self.K2 = nn.parameter.Parameter(torch.randn((N.shape[1], N.shape[2])), requires_grad=True)
            self.offset = nn.parameter.Parameter(torch.zeros((N.shape[1], N.shape[2])), requires_grad=True)

        if config.num_layers == 2:
            # self.encode = nn.ModuleList()
            # self.decode = nn.ModuleList()

            # for i in range(N.shape[1]):
            #     self.encode.append(nn.Linear(input_dim, hidden_dim))
            #     self.decode.append(nn.Linear(hidden_dim, input_dim))

            self.encode = nn.parameter.Parameter(torch.randn((N.shape[1], N.shape[2], hidden_dim)), requires_grad=True)
            self.encode_bias =  nn.parameter.Parameter(torch.zeros((N.shape[1], 1, hidden_dim)), requires_grad=True)
            self.decode = nn.parameter.Parameter(torch.randn((N.shape[1], hidden_dim, N.shape[2])), requires_grad=True)
            self.decode_bias =  nn.parameter.Parameter(torch.zeros((N.shape[1], N.shape[2])), requires_grad=True)

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

            z = torch.bmm(z, torch.bmm(self.decode, self.L)).squeeze()
            z = z + self.decode_bias
            return z

            # outputs = []

            # for i in range(self.N.shape[1]):
            #     z = self.encode[i](y[i])
            #     z = self.activation(z)
            #     z = self.decode[i](z)
            #     outputs.append(z)

            # return torch.stack(outputs)

    def forward_direct(self, t, y):
        # Calculate K_influx and K_outflux matrices (m, p, p)
        # K_influx should be the transpose of K_outflux in dim 1 and 2
        K_base = self.K[-1].unsqueeze(0)
        K_outflux = torch.cat([self.K[:-1] + K_base, K_base], dim=0)
        # Calculate dydt for each clone in range [0, n-1]
        y = y.unsqueeze(1)
        dydt = torch.bmm(y, K_outflux.transpose(1, 2)) - torch.bmm(y, K_outflux)

        return dydt.squeeze() # (m, p)