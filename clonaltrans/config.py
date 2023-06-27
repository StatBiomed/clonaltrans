class Configuration():
    def __init__(self):
        #* GENERAL SETTINGS *
        # TODO (int) speficy the GPU card, default 0, -1 will switch to CPU mode
        self.gpu = 3

        self.seed = 42

        #* MODEL ARCHETICTURE *
        # Avoiding non-smooth non-linearities such as ReLU and LeakyReLU 
        # and prefer non-linearities with a theoretically unique adjoint / gradient. 
        # Empirically, SoftPlus is better than GELU. NOT valid when self.K_type = 'const'
        self.activation = 'softplus'

        self.hidden_dim = 32

        # str, whether constant / dynamic / mixture / mixture_lr / K is used in model architecture, default const. 
        self.K_type = 'const'

        #* MODEL FITTING *
        # float, adjust penalty term on entries not in PAGA L (including diagonal) 
        # and upper bound of diagonal line, default 0.05
        self.alpha = 0.05
        # float, adjust penalty term on time-variant K(t), 0 -> no constraint
        # higher value -> K is more determined and resembles self.K_type = 'const', default 1e-3
        self.beta = 1e-3

        self.include_var = True

        #* OPTIMIZER & SCHEDULER
        self.learning_rate = 1e-3
        self.num_epochs = 1000
        self.lrs_ms = [200, 400, 600, 800]

        # Adjoint sensitivity method is used to compute gradients of the loss function w.r.t. the parameters of the ODE solver
        # For applications that require solving complex trajectories, recommend using the adjoint method
        # However, experimenting on small systems with direct backpropagation first is recommended
        # In our model, the adjoint method is roughly 2x~3x slower than non-adjoint backpropagation
        self.adjoint = False

        #* DATA PRE-PROCESSING *
        # parameter for specifing input data type to use, raw, log, root, shrink, etc.
        self.input_form = 'raw'

        self.exponent = 1. / 1.