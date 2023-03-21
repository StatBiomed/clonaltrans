from torch import nn
import torch

def activation_helper(activation):
    if activation == 'gelu':
        act = nn.GELU()
    if activation == 'sigmoid':
        act = nn.Sigmoid()
    if activation == 'tanh':
        act = nn.Tanh()
    if activation == 'relu':
        act = nn.ReLU()
    if activation == 'leakyrelu':
        act = nn.LeakyReLU()
    if activation is None:
        def act(x):
            return x
    return act

class ODEBlock(nn.Module):
    def __init__(self, N, input_dim, hidden_dim, activation: str = 'gelu'):
        super(ODEBlock, self).__init__()
        self.N = N
        self.activation = activation_helper(activation)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()

        for i in range(N.shape[1]):
            self.encode.append(nn.Linear(input_dim, hidden_dim))
            self.decode.append(nn.Linear(hidden_dim, input_dim))

    def forward(self, t, y):
        outputs = []

        for i in range(self.N.shape[1]):
            z = self.encode[i](y[i])
            z = self.activation(z)
            z = self.decode[i](z)
            outputs.append(z)

        return torch.stack(outputs)

    def forward_direct(self, t, y):
        # Calculate K_influx and K_outflux matrices (m, p, p)
        # K_influx should be the transpose of K_outflux in dim 1 and 2
        K_base = self.K[-1].unsqueeze(0)
        K_outflux = torch.cat([self.K[:-1] + K_base, K_base], dim=0)
        # Calculate dydt for each clone in range [0, n-1]
        y = y.unsqueeze(1)
        dydt = torch.bmm(y, K_outflux.transpose(1, 2)) - torch.bmm(y, K_outflux)

        return dydt.squeeze() # (m, p)