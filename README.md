# clonaltrans

Hi Melania, I've finished the code refactor process. Briefly, you only need to modify the parameters within the `./clonaltrans/config` folder.

I've provided two notebooks with comments for example analysis after getting the model, `Basic_Analysis_and_Bootstrapping.ipynb` and `Gillespie_Analysis.ipynb`.

Also there is a fitted example model located at `./trails/checkpoints/ClonalTransRates/0204_190019/model_last.pt`, which is generated using demo dataset at `./clonaltrans/demo_dataset` folder.

For **All** path parameters in `./clonaltrans/config` folder, please use **absolute path instead of relative path**.

## Basic Usage

To fit the model using clonal data,

``` python
python ./main.py --config ./config/main.json
```

Parameters within JSON file,

- K_type: 'dynamic' or 'const', whether transition rates are constant value
- alpha: penalize factor for upper bounds of rates and proliferation rate of fully differentiated populations
- beta: penalize factor for population with zero counts that have rates and std in GaussianNLL, **consider further increasing this parameter to suppress rates of zero counts population, might try 0.3 for instance?**
- no_proliferation_pops: please provide binary labels of fully differentiated populations which should NOT have any proliferation ability
- t_observed: please provide experimental time points
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

## Gillespie Algorithms (Stochastic Simulation)

To run Gillespie simulation given a model,

``` python
python ./main_gillespie.py --config ./config/main_gillespie.json
```

Parameters within JSON file,

- t_cutoff: we've noticed the Gillespie could run forever for certain circumstances, and this parameter controls the minimum time increment of the algorithm
