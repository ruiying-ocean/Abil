# import required packages
import pandas as pd
import numpy as np
import sys
from yaml import load
from yaml import CLoader as Loader
from planktonsdm.predict import predict
from sklearn.preprocessing import OneHotEncoder

try:
    print(sys.argv[1])
    with open('/user/work/ba18321/planktonSDM/configuration/2-phase_ensemble_cluster.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = True
    n_jobs = pd.to_numeric(sys.argv[1])
    n_spp = pd.to_numeric(sys.argv[2])
    root = model_config['hpc_root']
    model_config['cv'] = 10

except:
    with open('/user/work/ba18321/planktonSDM/configuration/2-phase_ensemble_cluster.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = False
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
d = d.dropna(subset=['FID'])

X_predict =  pd.read_csv(root + model_config['env_data_path'])
X_predict.set_index(["time", "depth", "lat", "lon"], inplace=True)
y = d[species]
X_train = d[predictors]

try:
    #drop rows with FIDs not found in original dataset:
    X_predict = X_predict[X_predict['FID'].isin(X_train['FID'])]

    enc = OneHotEncoder()
    regions  = enc.fit_transform(X_train[['FID']])
    X_train.loc[:, enc.categories_[0]]  = regions.toarray()

    X_train.columns = X_train.columns.astype(str)
    X_train = X_train.drop(columns=['FID'])

    regions  = enc.fit_transform(X_predict[['FID']])
    X_predict[enc.categories_[0]] = regions.toarray()
    X_predict.columns = X_predict.columns.astype(str)
    X_predict.loc[:, enc.categories_[0]]  = regions.toarray()
    X_predict = X_predict.drop(columns=['FID'])

except:
    print("FID not part of predictors")

#setup model:
m = predict(X_train, y, X_predict, model_config)
m.make_prediction()
