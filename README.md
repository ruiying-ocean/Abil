
# PlanktonSDM

## Overview

This repository provides functions to tune, predict, and post-process species distribution models using Python scikit-learn.

Generally the workflow is as follows:

0. Define the model setup in a model_config.yml file (for an example see `example_model_config`)
1. Tune the model for the species of interest using `tune.py`
2. Predict the distribution of each species using `predict.py`
3. Merge the predictions into a single netcdf and do post processing using `post.py`


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
|   └── ens
|       └── predictions
|       └── scoring
├── singularity
    └── Singularity.sif (optional)
    └── planktonSDM.sif (optional)

```

## Singularity

If you want to run the scripts on the cluster you need to create a container using Singularity: https://docs.sylabs.io/guides/3.4/user-guide/
To compile singularity run:

``` sudo singularity build planktonSDM.sif Singularity.sif  ```

## Transferring files to remote server

There are a few files in the distribution that are not needed to run the code on the cluster.

To upload only relevant files with scp there is a `to-bp.sh` bash script.

To run the script, cd to the planktonSDM directory:
``` cd path_to_planktonSDM ```

Set the bash script executable permission by running chmod command in Linux:

``` sudo chmod +x to-bp.sh ```

Then execute it:

``` sudo ./to-bp.sh ``` 

The script will ask for your cluster login information and then transfer the files.


