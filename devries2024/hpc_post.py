# import required packages
import sys
from yaml import load
from yaml import CLoader as Loader
from planktonsdm.post import post
import pandas as pd


try:
    print(sys.argv[0])
    with open('/user/work/ba18321/planktonSDM/devries2024/2-phase_ensemble_deVries2024.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = True
    root = model_config['hpc_root']

except:
    with open('/home/phyto/planktonSDM/devries2024/2-phase_ensemble_deVries2024.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = False
    root = model_config['local_root']


X_predict =  pd.read_csv(root + model_config['env_data_path'])
X_predict.set_index(["time", "depth", "lat", "lon"], inplace=True)


m = post(model_config)
m.merge_performance(model="ens")
m.merge_performance(model="xgb", configuration= "reg")
m.merge_performance(model="rf", configuration= "reg")
m.merge_performance(model="knn", configuration= "reg")

m.merge_parameters(model="rf")
m.merge_parameters(model="xgb")
m.merge_parameters(model="knn")

m.total()

m.merge_env(X_predict)

m.export_ds("17_oct")
m.export_csv("17_oct")
