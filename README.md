# CLADES

Repository for CLADES (Clonal Lineage Analysis with Differential Equations and Stochastic Simulations).

It contains the source code, and the datasets used in the paper, see `clonatrans` and `demo`. <br>
Notebooks with comments to reproduce the manuscript figures are provided within the `notebooks` folder. <br>
The configuration details used to train CLADES and the associtaed outputs are stored under `results`.

For **All** path parameters within `./clonaltrans/config` folder, please use **absolute path instead of relative path**.

## System Requirements

CLADES requires PyTorch, please make sure you have access to the GPUs. Other than that, we have provided a few essential packages that are required for the algorithm as well, listed in `requirements.txt`.

If there are verson conflicts within your own environment or for the reproducibility purpose, we also provide a complete list of dependencies and pip packages with specified version number that we used in our local environment, see `environments.yml`.

## Summary of Time Complexity

1. Data training stage, the estimated running time for CLADES is around 0.5~1 hour (depending on the size of the dataset and constant/dynamic modes) using 1 single GeForce RTX 3090 GPU.

2. Bootstrapping trials, as it's a multi-processing module that works in parallel, the actual running time varies depending on the number/performance of GPU cards used. Empirically, when using 4 cards to bootstrap 800 times, the expected running time is around 10 hours.

3. Gillespie simulations, it uses CPU only. Though multi-processing technique has been adpoted to speed up the process, the estimated running time is still a few hours (for 1,000 simulations as demonstrated in the manuscript).

## Installation

No installation process is needed for CLADES, please clone the repository locally then it is ready to use.

```bash
git clone https://github.com/StatBiomed/clonaltrans.git
```

## Basic Usage

To fit the model using clonal data, 

1) Data preparation: 

    We have provided the input files for the datasets used by CLADES, each has 4 separate files. Please follow the format given in the `demo` folder.

    If you would like to test your own dataset, the pipeline for generating those files can be found via `./demo/CordBlood_Refine/prepare_input.ipynb` or `./demo/Weinreb/prepare_input.ipynb`.
    
2) Model configuration:

    The configuration files (JSON format) for CLADES are located at `./clonaltrans/config` folder. 

    We have provided here all the config files for both the human cord blood and mouse hematopoiesis dataset, for all analysis: `constant mode` `dynamic mode` `bootstrapping` and `gillespie analysis`, as shown on file names. 

    To run CLADES, please execute the following command, e.g.,

    ``` python
    python ./main.py --config ./config/main_dynamic_cordblood.json
    ```

For full list of tunable parameters, please refer to the JSON file, here are a few commonly used parameters,

- K_type: 'dynamic' or 'const', whether transition rates are constant value
- alphas: coefficients of the penalties
- no_proliferation_pops: please provide binary labels of fully differentiated populations which should NOT have strong proliferation ability, e.g., terminal states
- no_apoptosis_pops: please provide binary labels of non differentiated populations which should have strong proliferation ability, e.g., early progenitors
- t_observed: please provide real experimental time points
- scaling_facotr: please provide the scaling factor to total counts for each time points 
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

## Gillespie Algorithms (Stochastic Simulation)

To run Gillespie simulation given a model,

``` python
python ./main_gillespie.py --config ./config/main_gillespie.json
```

Parameters within JSON file,

- t_cutoff: we've noticed the Gillespie could run forever for certain circumstances, and this parameter controls the minimum time increment of the algorithm