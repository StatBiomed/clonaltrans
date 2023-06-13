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
        act = nn.Softplus(threshold=7)
    if activation is None:
        def act(x):
            return x
    return act

class ODEBlock(nn.Module):
    def __init__(
        self, 
        num_tpoints: int = 4,
        num_clones: int = 11,
        num_pops: int = 11,
        hidden_dim: int = 16, 
        activation: str = 'softplus', 
        num_layers: int = 1
    ):
        super(ODEBlock, self).__init__()
        self.nfe = 0
        self.num_layers = num_layers
        self.std = torch.ones((1, num_clones, num_pops), dtype=torch.float32) * 1
        self.std = Parameter(self.std, requires_grad=True)
        self.activation = activation_helper(activation)

        if self.num_layers == 1:
            '''
            ODE function dydt = K1 * y + K2 * y + bias (optional) with KaiMing Initialization
            K1, >= 0, upper off-diagonal, other populations flow into designated population
            K2, diagonal, self proliferation and cell apoptosis process
            K3, >= 0, should be transpose of K1, lower off-diagonal
            ''' 

            self.K1, self.K2 = [], []

            for clone in range(num_clones):
                K1 = torch.empty((num_pops, num_pops))
                K2 = torch.empty((1, num_pops))

                self.reset_parameters(K1, None)
                self.reset_parameters(K2, None)

                self.K1.append(K1.T)
                self.K2.append(K2.T)

            self.K1 = Parameter(torch.stack(self.K1), requires_grad=True)
            self.K2 = Parameter(torch.stack(self.K2).squeeze(), requires_grad=True)

            self.K1_mask = torch.triu(torch.ones((self.K1.shape[1], self.K1.shape[2])), diagonal=1)
            self.K1_mask = Parameter(torch.broadcast_to(self.K1_mask.unsqueeze(0), self.K1.shape), requires_grad=False)

        if self.num_layers == 2:
            #* Batch processing 2 layer MLP for each individual clone with KaiMing Initialization
            self.encode, self.encode_bias = [], None
            self.decode, self.decode_bias = [], None

            # self.encode, self.encode_bias = self.linear_init(num_clones, num_pops, hidden_dim)
            # self.decode, self.decode_bias = self.linear_init(num_clones, hidden_dim, num_pops)

            for clone in range(num_clones):
                encode = torch.empty((hidden_dim, num_pops))
                decode = torch.empty((num_pops, hidden_dim))

                self.reset_parameters(encode, None)
                self.reset_parameters(decode, None)

                self.encode.append(encode.T)
                self.decode.append(decode.T)
            
            self.encode = Parameter(torch.stack(self.encode), requires_grad=True)
            self.decode = Parameter(torch.stack(self.decode), requires_grad=True)
        
        if self.num_layers == 3:
            self.k1_enw, self.k1_enb = self.linear_init(num_clones, num_pops, 32)
            self.k1_dew, self.k1_deb = self.linear_init(num_clones, 32, num_pops * num_pops)
            self.k2_enw, self.k2_enb = self.linear_init(num_clones, num_pops, 32)
            self.k2_dew, self.k2_deb = self.linear_init(num_clones, 32, num_pops)

            self.K1_mask = Parameter(torch.triu(torch.ones((num_pops, num_pops)), diagonal=1).unsqueeze(0), requires_grad=False)

    def linear_init(self, num_clones, in_dim, out_dim):
        weight_matrix, bias_matrix = [], []

        for clone in range(num_clones):
            weight = torch.empty((out_dim, in_dim))
            bias = torch.empty((out_dim, ))

            self.reset_parameters(weight, bias)

            weight_matrix.append(weight.T)
            bias_matrix.append(bias)

        return Parameter(torch.stack(weight_matrix), requires_grad=True), Parameter(torch.stack(bias_matrix), requires_grad=True)

    def get_K1_K2(self, y):
        z1 = torch.bmm(y.unsqueeze(1), self.k1_enw)
        z1 = self.activation(z1)

        z1 = torch.bmm(z1, self.k1_dew)
        z1 = torch.square(z1.squeeze().view(y.shape[0], y.shape[1], y.shape[1])) * self.K1_mask

        z2 = torch.bmm(y.unsqueeze(1), self.k2_enw)
        z2 = self.activation(z2)

        z2 = torch.bmm(z2, self.k2_dew)
        return z1, z2.squeeze()

    def forward(self, t, y):
        self.nfe += 1
        # print (t)

        if self.num_layers == 1:
            z = torch.bmm(y.unsqueeze(1), torch.square(self.K1) * self.K1_mask).squeeze()
            z = z + y * self.K2 
            z = z - torch.sum(y.unsqueeze(1) * torch.square(self.K1.mT) * self.K1_mask.mT, dim=1)

            return z
        
        if self.num_layers == 2:
            z = torch.bmm(y.unsqueeze(1), self.encode)
            z = self.activation(z)
            z = torch.bmm(z, self.decode).squeeze()
            return z
    
        if self.num_layers == 3:
            k1, k2 = self.get_K1_K2(y)
            z = torch.bmm(y.unsqueeze(1), k1).squeeze()
            z = z + y * k2 
            z = z - torch.sum(y.unsqueeze(1) * k1.mT, dim=1)
            return z

    def reset_parameters(self, weight, bias) -> None:
        # nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        # nn.init.xavier_normal_(weight)
        nn.init.normal_(weight, 0, 0.01)
        # nn.init.kaiming_normal_(weight)

        if bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias, -bound, bound)

    def extra_repr(self) -> str:
        if self.num_layers == 1:
            return 'K1 (clone, pop, pop) = {}, \nK2 (clone, pop) = {}'.format(
                self.K1.shape, self.K2.shape
            )
        
        if self.num_layers == 2:
            return 'encode (clone, pop, hidden) = {}, \nencode_bias={}, \ndecode (clone, hidden, pop) = {}, \ndecode_bias={}'.format(
                self.encode.shape, self.encode_bias is not None, self.decode.shape, self.decode_bias is not None
            )

        if self.num_layers == 3:
            expr1 = 'K1_encode (clone, pop, hidden) = {}, \nencode_bias={}, \nK1_decode (clone, hidden, pop * pop) = {}, \ndecode_bias={}'.format(
                self.k1_enw.shape, self.k1_enb is not None, self.k1_dew.shape, self.k1_deb is not None
            )

            expr2 = 'K2_encode (clone, pop, hidden) = {}, \nencode_bias={}, \nK2_decode (clone, hidden, pop) = {}, \ndecode_bias={}'.format(
                self.k2_enw.shape, self.k2_enb is not None, self.k2_dew.shape, self.k2_deb is not None
            )

            return expr1 + '\n' + expr2