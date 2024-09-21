
import itertools
import numpy as np

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
            return bool(obj)
    else:
        return obj

def get_base_combinations():
    tunablereturn_params = {
        'seed': np.random.randint(20, size=1)[0],
        'hidden_dim': np.random.choice([16, 32, 64], size=1)[0],
        'alphas': [
            np.round(np.random.choice(np.arange(0, 1.1, 0.1), size=1)[0], 1),
            np.round(np.random.choice(np.arange(0, 1.1, 0.1), size=1)[0], 1),
            np.round(np.random.choice(np.arange(0, 1.1, 0.1), size=1)[0], 1),
            np.round(np.random.choice(np.arange(0, 1.1, 0.1), size=1)[0], 1),
            np.round(np.random.choice(np.arange(0, 1.1, 0.1), size=1)[0], 1),
            np.round(np.random.choice(np.arange(0, 1.1, 0.1), size=1)[0], 1)
        ],
        'weighted_rate': np.random.choice([True, False], size=1)[0],
        'leiden_resolution': np.round(np.random.choice(np.arange(0.5, 3.0, 0.1), size=1)[0], 1),
    }
    return tunablereturn_params

def get_nni_leiden(resolution):
    tunable_params = {
        'seed': 42,
        'hidden_dim': 32,
        'alphas': [
            1.0,
            0.5,
            0.5,
            0.5,
            0.1,
            1.0
        ],
        'weighted_rate': True,
        'leiden_resolution': resolution,
    }
    return tunable_params

def get_nni_seeds(seed):
    tunable_params = {
        'seed': seed,
        'hidden_dim': 32,
        'alphas': [
            1.0,
            0.5,
            0.5,
            0.5,
            0.1,
            1.0
        ],
        'weighted_rate': True,
        'leiden_resolution': 1.0,
    }
    return tunable_params

def get_nni_hidden_dims(hidden_dim):
    tunable_params = {
        'seed': 42,
        'hidden_dim': hidden_dim,
        'alphas': [
            1.0,
            0.5,
            0.5,
            0.5,
            0.1,
            1.0
        ],
        'weighted_rate': True,
        'leiden_resolution': 1.0,
    }
    return tunable_params

def get_nni_alphas():
    tunable_params = {
        'seed': 42,
        'hidden_dim': 32,
        'alphas': [
            np.round(np.random.choice(np.arange(0.1, 1.1, 0.1), size=1)[0], 1),
            np.round(np.random.choice(np.arange(0.1, 1.1, 0.1), size=1)[0], 1),
            np.round(np.random.choice(np.arange(0.1, 1.1, 0.1), size=1)[0], 1),
            np.round(np.random.choice(np.arange(0.1, 1.1, 0.1), size=1)[0], 1),
            np.round(np.random.choice(np.arange(0.1, 1.1, 0.1), size=1)[0], 1),
            np.round(np.random.choice(np.arange(0.1, 1.1, 0.1), size=1)[0], 1)
        ],
        'weighted_rate': True,
        'leiden_resolution': 0.25,
    }
    return tunable_params

def get_nni_architecture(option=2):
    if option == 1:
        tunable_params = {
            'clipping': np.random.choice([True, False], size=1)[0],
            'weighted_rate': np.random.choice([True, False], size=1)[0],
            'scheduler_type': np.random.choice(["AutoAdaptive", "MultiStepLR"], size=1)[0],
        }
    
    if option == 2:
        param_options = {
            'clipping': [True, False],
            'weighted_rate': [True, False],
            'scheduler_type': ["AutoAdaptive", "MultiStepLR"],
        }

        keys = param_options.keys()
        combinations = list(itertools.product(*param_options.values()))
        tunable_params = [dict(zip(keys, combination)) for combination in combinations]

    return tunable_params

def get_nni_combinations():
    return get_nni_architecture(option=2)

def prepare_nni(config, tunable_params):
    # config['system']['seed'] = tunable_params['seed']
    # config['arch']['args']['hidden_dim'] = tunable_params['hidden_dim']
    # config['user_trainer']['alphas'] = tunable_params['alphas']
    # config['user_trainer']['weighted_rate'] = tunable_params['weighted_rate']
    # config['nni_data']['leiden_resolution'] = tunable_params['leiden_resolution']

    config['arch']['args']['clipping'] = tunable_params['clipping']
    config['user_trainer']['weighted_rate'] = tunable_params['weighted_rate']
    config['optimizer']['scheduler_type'] = tunable_params['scheduler_type']

    return config

