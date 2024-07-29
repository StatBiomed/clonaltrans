# CLADES

Repository for CLADES (Clonal Lineage Analysis with Differential Equations and Stochastic Simulations).

It contains the source code, demo data and key results used in the paper. Notebooks with comments are provided within the `results` folder. Also the fitted model and configuration details are provided as well.

For **All** path parameters in `./clonaltrans/config` folder, please use **absolute path instead of relative path**.

To reproduce the figures presented in the manuscript, please refer to the `results` folder. We provide separate notebooks for each analysis.

## System Requirements

CLADES was tested on Linux platform both with command lines and Jupyter Notebooks.

The basic Python packages required for the algorithm is listed in `requirements.txt`.

If verson conflicts exist or you are unable to run the alogirhm on your own environment, we also provide a complete list of dependencies and pip packages with specified version number in `environments.yml`.

All estimated running time mentioned here is based on 1 single GeForce RTX 3090 GPU, if you only have CPUs, the expected running time could be a few folds longer.

## Installation

No installation process is needed for CLADES, please clone the repository locally then it is ready to use.

```bash
git clone https://github.com/StatBiomed/clonaltrans.git
```

## Basic Usage

To fit the model using clonal data, 

1) Data preparation: 

    Please follow the format given in the `demo` folder. Example pipeline to prepare the data are located at `./demo/CordBlood_Refine/prepare_input.ipynb`.

    You could directly use data within `demo` folder or use your own data to test the algorithm.
    
2) Model configuration:

    Only parameters within the `./clonaltrans/config` folder need to be modified. The parameters are stored in JSON format, and CLADES can be executed with command lines,

    ``` python
    python ./main.py --config ./config/main.json
    ```

    Normally the runtime is between 30mins to 1h, depending on the used modes.

For full list of tunable parameters, please refer to the JSON file, here are a few commonly used parameters,

- K_type: 'dynamic' or 'const', whether transition rates are constant value
- alphas: coefficients of the penalties
- no_proliferation_pops: please provide binary labels of fully differentiated populations which should NOT have strong proliferation ability
- no_apoptosis_pops: please provide binary labels of non differentiated populations which should have strong proliferation ability
- t_observed: please provide real experimental time points
- scaling_facotr: please provide the scaling factor to total counts for each time points 
- ub_for_prol: upper bound for per capita proliferation rates
- ub_for_diff: upper bound for per capita differentiation rates
- learning_rate: 1e-3 for dynamic mode and 5e-2 for const mode

## Bootstrapping for Confidence Intervals

To estimate the confidence intervals for each transition rate using bootstrapping method,

``` python
python ./main_bootstrap.py --config ./config/main_bootstrap.json
```

Parameters within JSON file,

- gpu_id: bootstrapping could use multiprocessing if there's multiple GPUs available, please provide the GPU index in a list format
- model_path: absolute path of fitted model generated in `Basic Usage`
- concurrent: # of boostraps to perform at the same time
- epoch: # of epochs, for instance, concurrent 5 and epochs 60 means in total the model will be bootstrapped for 300 times

The runtime is approximately a few hours, depending on number of bootstrap trails you need.

## Gillespie Algorithms (Stochastic Simulation)

To run Gillespie simulation given a model,

``` python
python ./main_gillespie.py --config ./config/main_gillespie.json
```

Parameters within JSON file,

- t_cutoff: we've noticed the Gillespie could run forever for certain circumstances, and this parameter controls the minimum time increment of the algorithm

Usually this process is done within 1 hour.