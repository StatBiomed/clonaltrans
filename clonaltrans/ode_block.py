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
            self.activation = activation_helper(activation)

            for clone in range(num_clones):
                encode = torch.empty((hidden_dim, num_pops))
                decode = torch.empty((num_pops, hidden_dim))

                self.reset_parameters(encode, None)
                self.reset_parameters(decode, None)

                self.encode.append(encode.T)
                self.decode.append(decode.T)
            
            self.encode = Parameter(torch.stack(self.encode), requires_grad=True)
            self.decode = Parameter(torch.stack(self.decode), requires_grad=True)

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
        if self.num_layers == 1:
            return 'K1 (clone, pop, pop) = {}, \nK2 (clone, pop) = {}'.format(
                self.K1.shape, self.K2.shape
            )
        
        if self.num_layers == 2:
            return 'encode (clone, pop, hidden) = {}, \nencode_bias={}, \ndecode (clone, hidden, pop) = {}, \ndecode_bias={}'.format(
                self.encode.shape, self.encode_bias is not None, self.decode.shape, self.decode_bias is not None
            )

    def forward(self, t, y):
        self.nfe += 1

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

    def forward_direct(self, t, y):
        # Calculate K_influx and K_outflux matrices (m, p, p)
        # K_influx should be the transpose of K_outflux in dim 1 and 2
        K_base = self.K[-1].unsqueeze(0)
        K_outflux = torch.cat([self.K[:-1] + K_base, K_base], dim=0)
        # Calculate dydt for each clone in range [0, n-1]
        y = y.unsqueeze(1)
        dydt = torch.bmm(y, K_outflux.transpose(1, 2)) - torch.bmm(y, K_outflux)

        return dydt.squeeze() # (m, p)

######################
# So you want to train a Neural CDE model?
# Let's get started!
######################

import math
import torch
import torchcde


######################
# A CDE model looks like
#
# z_t = z_0 + \int_0^t f_\theta(z_s) dX_s
#
# Where X is your data and f_\theta is a neural network. So the first thing we need to do is define such an f_\theta.
# That's what this CDEFunc class does.
# Here we've built a small single-hidden-layer neural network, whose hidden layer is of width 128.
######################
class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()
        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


######################
# Next, we need to package CDEFunc up into a model that computes the integral.
######################
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, interpolation="cubic"):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation

    def forward(self, coeffs):
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        ######################
        # Actually solve the CDE.
        ######################
        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=X.interval)

        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value,
        # and then apply a linear map.
        ######################
        z_T = z_T[:, 1]
        pred_y = self.readout(z_T)
        return pred_y


######################
# Now we need some data.
# Here we have a simple example which generates some spirals, some going clockwise, some going anticlockwise.
######################
def get_data(num_timepoints=100):
    t = torch.linspace(0., 4 * math.pi, num_timepoints)

    start = torch.rand(128) * 2 * math.pi
    x_pos = torch.cos(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos[:64] *= -1
    y_pos = torch.sin(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos += 0.01 * torch.randn_like(x_pos)
    y_pos += 0.01 * torch.randn_like(y_pos)
    ######################
    # Easy to forget gotcha: time should be included as a channel; Neural CDEs need to be explicitly told the
    # rate at which time passes. Here, we have a regularly sampled dataset, so appending time is pretty simple.
    ######################
    X = torch.stack([t.unsqueeze(0).repeat(128, 1), x_pos, y_pos], dim=2)
    y = torch.zeros(128)
    y[:64] = 1

    perm = torch.randperm(128)
    X = X[perm]
    y = y[perm]

    ######################
    # X is a tensor of observations, of shape (batch=128, sequence=100, channels=3)
    # y is a tensor of labels, of shape (batch=128,), either 0 or 1 corresponding to anticlockwise or clockwise
    # respectively.
    ######################
    return X, y


def main(num_epochs=30):
    train_X, train_y = get_data()

    ######################
    # input_channels=3 because we have both the horizontal and vertical position of a point in the spiral, and time.
    # hidden_channels=8 is the number of hidden channels for the evolving z_t, which we get to choose.
    # output_channels=1 because we're doing binary classification.
    ######################
    model = NeuralCDE(input_channels=3, hidden_channels=8, output_channels=1)
    optimizer = torch.optim.Adam(model.parameters())

    ######################
    # Now we turn our dataset into a continuous path. We do this here via Hermite cubic spline interpolation.
    # The resulting `train_coeffs` is a tensor describing the path.
    # For most problems, it's probably easiest to save this tensor and treat it as the dataset.
    ######################
    train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_X)

    train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch_coeffs, batch_y = batch
            pred_y = model(batch_coeffs).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()))

    test_X, test_y = get_data()
    test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_X)
    pred_y = model(test_coeffs).squeeze(-1)
    binary_prediction = (torch.sigmoid(pred_y) > 0.5).to(test_y.dtype)
    prediction_matches = (binary_prediction == test_y).to(test_y.dtype)
    proportion_correct = prediction_matches.sum() / test_y.size(0)
    print('Test Accuracy: {}'.format(proportion_correct))


if __name__ == '__main__':
    main()