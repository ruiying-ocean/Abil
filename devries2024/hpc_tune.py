# import required packages
import pandas as pd
import numpy as np
import sys
from yaml import load
from yaml import CLoader as Loader
from abil.tune import tune 
from sklearn.preprocessing import OneHotEncoder

try:
    print(sys.argv[1])
    with open('/user/work/ba18321/planktonSDM/devries2024/ensemble_regressor_deVries2024.yml', 'r') as f:
        model_config = load(f, Loader=Loader)

    model_config['hpc'] = True
    n_jobs = pd.to_numeric(sys.argv[1])
    n_spp = pd.to_numeric(sys.argv[2])
    root = model_config['hpc_root']
    model_config['cv'] = 10
    model = sys.argv[3]
    predictors = model_config['predictors']

except:
    with open('/home/phyto/planktonSDM/devries2024/ensemble_regressor_deVries2024.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = False
    n_jobs = 8
    n_spp = 1
    root = model_config['local_root']
    model_config['cv'] = 3
    
    with open('/home/phyto/planktonSDM/devries2024/ensemble_regressor_deVries2024.yml', 'r') as f:
        model_config_local = load(f, Loader=Loader)    
    
    model_config['param_grid'] = model_config_local['param_grid'] 
    model = "rf"


#define model config:
model_config['n_threads'] = n_jobs
traits = pd.read_csv(root + model_config['traits'])
d = pd.read_csv(root + model_config['training'])
species =  traits['species'][n_spp]
d = d.dropna(subset=[species])
d = d.dropna(subset=['FID'])

y = d[species]
X_train = d[predictors]

#setup model:
m = tune(X_train, y, model_config, regions="FID")
#run model:
m.train(model=model, regressor=True, log="both")
