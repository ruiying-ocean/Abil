# import required packages
import sys
from yaml import load
from yaml import CLoader as Loader
from planktonsdm.post import post

try:
    print(sys.argv[0])
    with open('/user/work/ba18321/planktonSDM/configuration/2-phase_ensemble_cluster.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = True

except:
    with open('/home/phyto/planktonSDM/configuration/2-phase_ensemble_cluster.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = False


m = post(model_config)
m.merge_performance(model="ens")
m.merge_performance(model="xgb", configuration= "reg")
m.merge_performance(model="xgb", configuration= "zir")
m.merge_performance(model="rf", configuration= "reg")
m.merge_performance(model="knn", configuration= "reg")
m.merge_performance(model="rf", configuration= "zir")
m.merge_performance(model="knn", configuration= "zir")

m.merge_parameters(model="rf")
m.merge_parameters(model="xgb")
m.merge_parameters(model="knn")

m.total()

m.merge_env()

m.export_ds("17_oct")
m.export_csv("17_oct")
