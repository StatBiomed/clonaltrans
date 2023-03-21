class Configuration():
    def __init__(self):
        #* GENERAL SETTINGS *
        # TODO (int) speficy the GPU card, default 0, -1 will switch to CPU mode
        self.gpu = 3

        self.seed = 42

        #* MODEL ARCHETICTURE *
        self.activation = 'gelu'

        #* MODEL FITTING *
        self.alpha = 0.01
        
        self.beta = 0.01

        self.num_epochs = 3000
        
        self.learning_rate = 0.01

        #* DATA PRE-PROCESSING *
        self.paga_diagonal = True
        