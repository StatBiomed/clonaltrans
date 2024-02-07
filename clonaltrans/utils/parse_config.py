import logging
from pathlib import Path
from functools import partial
from datetime import datetime
from logger import setup_logging
from utils.utility import read_json, write_json

class ConfigParser:
    def __init__(self, config, run_id=None):
        self.config = config
        self.run_id = run_id

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['base_trainer']['save_dir'])    
        exper_name = Path(self.config['name'])

        # use timestamp as default run-id
        if self.run_id is None: 
            self.run_id = datetime.now().strftime(r'%m%d_%H%M%S')

        self._save_dir = save_dir / 'checkpoints' / exper_name / self.run_id
        self._log_dir = save_dir / 'datasplit' / exper_name / self.run_id
        print('save_dir', self._save_dir)    

        # make directory for saving checkpoints and log.
        self.save_dir.mkdir(parents=True, exist_ok=False)
        self.log_dir.mkdir(parents=True, exist_ok=False)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')    

        setup_logging(self.log_dir, self.config['logger_config_path'])   
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }    

    @classmethod
    def from_args(cls, args):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        if not isinstance(args, tuple):
            args = args.parse_args()
        
        if args.run_id is not None:
            run_id = args.run_id
        else:
            run_id = None

        msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
        assert args.config is not None, msg_no_cfg

        cfg_fname = Path(args.config)
        config = read_json(cfg_fname)
        
        return cls(config, run_id)    

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.
        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.
        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir       
