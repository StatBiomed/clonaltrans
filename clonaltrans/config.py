class Configuration():
    def __init__(self):
        #* GENERAL SETTINGS *
        # TODO (int) speficy the GPU card, default 0, -1 will switch to CPU mode
        self.gpu = 3

        self.seed = 42

        #* MODEL ARCHETICTURE *
        # Avoiding non-smooth non-linearities such as ReLU and LeakyReLU 
        # and prefer non-linearities with a theoretically unique adjoint / gradient. 
        # Empirically, SoftPlus is better than GELU. NOT valid when self.num_layers = 1
        self.activation = 'softplus'

        self.hidden_dim = 16

        # int, number of linear layers used in model architecture, default 2. 
        # self.num_layers = 1 also supported, whilst performance not ideal
        self.num_layers = 2

        #* MODEL FITTING *
        # float, parameter to adjust the magnitude of loss on Δ, default 0.01
        self.alpha = 0.01
        # float, parameter to adjust the magnitude of loss on base K, default 0.01
        self.beta = 0.01

        self.num_epochs = 1000
        
        # optimizer & scheduler related
        self.learning_rate = 0.01
        self.lrs_ms = [300, 500, 600, 700, 800]
        self.lrs_gamma = 0.5

        # Adjoint sensitivity method is used to compute gradients of the loss function w.r.t. the parameters of the ODE solver
        # For applications that require solving complex trajectories, recommend using the adjoint method
        # However, experimenting on small systems with direct backpropagation first is recommended
        # In our model, the adjoint method is roughly 2x~3x slower than non-adjoint backpropagation
        self.adjoint = False

        #* DATA PRE-PROCESSING *
        # parameter for specifing input data type to use, raw, log, root, shrink, etc.
        self.input_form = 'root'