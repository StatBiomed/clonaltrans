class Configuration():
    def __init__(self):
        #* GENERAL SETTINGS *
        # TODO (int) speficy the GPU card, default 0, -1 will switch to CPU mode
        self.gpu = 3

        self.seed = 42

        #* MODEL ARCHETICTURE *
        self.activation = 'softplus'

        # self.hidden_dim = 16

        self.num_layers = 1

        #* MODEL FITTING *
        self.alpha = 0.01
        
        self.beta = 0.01

        self.num_epochs = 1500
        
        self.learning_rate = 0.01

        self.lrs_ms = [200, 300, 400, 500, 600, 700]

        self.lrs_gamma = 0.5

        self.adjoint = False

        #* DATA PRE-PROCESSING *
        # self.paga_diagonal = True

        # self.init_day_zero = True
        
        self.simulation = False

        # TODO parameter for specifing input data type to use, raw, log, etc.
        # self.log_data = False