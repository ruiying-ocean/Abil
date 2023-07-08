# import required packages
import pandas as pd
import numpy as np
import sys
from yaml import load
from yaml import CLoader as Loader

try:
    print(sys.argv[1])
    sys.path.insert(0, '/user/work/ba18321/planktonSDM/functions/')
    from tune import tune 

    with open('/user/work/ba18321/planktonSDM/configuration/devries2023_model_config.yml', 'r') as f:
        model_config = load(f, Loader=Loader)

    model_config['remote'] = True
    n_jobs = pd.to_numeric(sys.argv[1])
    n_spp = pd.to_numeric(sys.argv[2])
    model_config['root'] = model_config['remote_root']
    model_config['cv'] = 10
    model_config['param_grid'] = model_config['remote_param_grid'] 

except:
    sys.path.insert(0, '/home/phyto/planktonSDM/functions/')
    from tune import tune 
    with open('/home/phyto/planktonSDM/configuration/devries2023_model_config.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['remote'] = False
    n_jobs = 8
    n_spp = 1
    model_config['root'] = model_config['local_root']
    model_config['cv'] = 3
    model_config['param_grid'] = model_config['local_param_grid'] 

#define model config:
model_config['n_threads'] = n_jobs
traits = pd.read_csv(model_config['root'] + model_config['traits'])
d = pd.read_csv(model_config['root']+ model_config['training'])
species =  traits['species'][n_spp]
predictors = model_config['predictors']
d = d.dropna(subset=[species])

y = d[species]
X = d[predictors]

#setup model:
m = tune(X, y, model_config)
#run model:
m.train(model="xgb", zir=True, log="both")