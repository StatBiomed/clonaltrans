class Configuration():
    def __init__(self):
        #* GENERAL SETTINGS *
        # TODO (int) speficy the GPU card, default 0, -1 will switch to CPU mode
        self.gpu = 3

        self.seed = 42

        self.inspect = True

        #* MODEL ARCHETICTURE *
        self.activation = 'gelu'

        self.hidden_dim = 16

        #* MODEL FITTING *
        self.alpha = 0.01
        
        self.beta = 0.01

        self.num_epochs = 2000
        
        self.learning_rate = 0.01

        self.lrs_step = 200

        self.lrs_gamma = 0.5

        # TODO parameter for specifing input data type to use, raw, log, etc.
        self.log_data = True

        #* DATA PRE-PROCESSING *
        self.paga_diagonal = True

        self.init_day_zero = True
        