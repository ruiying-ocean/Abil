# import required packages
import pandas as pd
import numpy as np
import sys
from yaml import load
from yaml import CLoader as Loader
from planktonsdm.predict import predict

try:
    print(sys.argv[1])
    with open('/user/work/ba18321/planktonSDM/configuration/2-phase_ensemble_cluster.yml', 'r') as f:
        model_config = load(f, Loader=Loader)

    model_config['remote'] = True
    n_jobs = pd.to_numeric(sys.argv[1])
    n_spp = pd.to_numeric(sys.argv[2])
    root = model_config['hpc_root']
    model_config['cv'] = 10

except:
    with open('/user/work/ba18321/planktonSDM/configuration/2-phase_ensemble_cluster.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['remote'] = False
    n_jobs = 8
    n_spp = 1
    root = model_config['local_root']
    model_config['cv'] = 3


#define model config:
model_config['n_threads'] = n_jobs
traits = pd.read_csv(root + model_config['traits'])
d = pd.read_csv(root + model_config['training'])
species =  traits['species'][n_spp]
predictors = model_config['predictors']
d = d.dropna(subset=[species])
envdata =  pd.read_csv(root + model_config['env_data_path'])
    
y = d[species]
X = d[predictors]

#setup model:
m = predict(X, y, envdata, model_config)
m.make_prediction()
