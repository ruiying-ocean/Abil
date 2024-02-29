
# ABIL (Aquatic Biogeochemical Interpolation Library)

## Overview

ABIL provides functions to tune, predict, and post-process distributions of biogeochemical observations using Python scikit-learn and MAPIE. The library is optimized to interpolate many predictions in parallel and is thus particularly suited for distribution models of species, genes and transcripts. 

Current support (v0.08):

- Random Forest, XGBoost, Bagged KNN, and MultiLayer Perceptron

- Presence/Absence and continuous data

- Automatic feature scaling and one-hot-encoding

- Model error prediction inference (MAPIE)

- SLURM and Singularity scripts


Generally the workflow is as follows:

1. Define the model setup in a model_config.yml file (for an example see `example_model_config`)
2. Tune the model for the species of interest using `tune.py`
3. Predict the distribution of each species using `predict.py`
4. Merge the predictions into a single netcdf and do post processing using `post.py`


Examples for each step are provided in the respective Jupyter notebooks which can be found in `/notebooks`.


## Directory structure

The recommended directory structure is:


```bash

planktonSDM
├── README.md
├── to-bp.sh
├── environment.yml
├── configuration
|   └── model_config.yml
├── data
|   └── traits.csv
|   └── abundances_environment.csv (optional)
|   └── envdata.csv (optional)
├── planktonsdm
|   └── __init__.py
|   └── functions.py
|   └── post.py
|   └── predict.py
|   └── tune.py
├── ModelOutput
|   └── rf
|       └── model
|       └── predictions
|       └── scoring
|   └── knn
|       └── model
|       └── predictions
|       └── scoring
|   └── xgb
|       └── model
|       └── predictions
|       └── scoring
|   └── mlp
|       └── model
|       └── predictions
|       └── scoring
|   └── ens
|       └── predictions
|       └── scoring
└── singularity
    └── Singularity.sif (optional)
    └── planktonSDM.sif (optional)

```

## Installing the package:

Install the dependencies in a new environment: 

``` conda env create -f package_save_path/environment.yml ``` 

Activate the new environment and install planktonSDM:

``` conda activate planktonsdm-env ``` 

``` python -m pip install package_save_path  ``` 

## Updating the package:

If you have changed the scripts and want to update the package, a new version can be build.

CD to the planktonSDM directory, then run:

``` python3 -m build  ``` 

Note: if you want to change the version name of the package, this can be changed in:

`planktonSDM/pyproject.toml`

## Running the model on a hpc cluster

See: `hpc_example/README.md`