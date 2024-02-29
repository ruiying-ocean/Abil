# import required packages
import sys
from yaml import load
from yaml import CLoader as Loader
from abil.post import post
import pandas as pd

from datetime import datetime
current_date = datetime.today().strftime('%Y-%m-%d')


try:
    print(sys.argv[0])
    with open('/user/work/ba18321/abil/devries2024/ensemble_regressor_deVries2024.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = True
    root = model_config['hpc_root']

except:
    with open('/home/phyto/planktonSDM/devries2024/ensemble_regressor_deVries2024.yml', 'r') as f:
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
m.export_ds(current_date + "_ci50")
m.export_csv(current_date + "_ci50")

m = post(model_config, ci=32)
m.total()
m.merge_env(X_predict)
m.export_ds(current_date + "_ci32")
m.export_csv(current_date + "_ci32")

m = post(model_config, ci=68)
m.total()
m.merge_env(X_predict)
m.export_ds(current_date + "_ci68")
m.export_csv(current_date + "_ci68")

