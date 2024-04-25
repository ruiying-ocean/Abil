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
    with open('/user/work/ba18321/Abil/devries2024/2-phase.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = True
    root = model_config['hpc_root']

except:
    with open('/home/phyto/Abil/devries2024/2-phase.yml', 'r') as f:
        model_config = load(f, Loader=Loader)
    model_config['hpc'] = False
    root = model_config['local_root']


X_predict =  pd.read_csv(root + model_config['env_data_path'])
X_predict.set_index(["time", "depth", "lat", "lon"], inplace=True)

print("path:")
print(root + model_config['targets'])
targets = pd.read_csv(root + model_config['targets'])
#print("targets: ")
#print(targets)
targets =  targets['Target'].values
depth_w = 5
conversion = 1e3 #L-1 to m-3

m = post(model_config)
m.merge_performance(model="ens")
m.merge_performance(model="xgb", configuration= "reg")
m.merge_performance(model="rf", configuration= "reg")
m.merge_performance(model="knn", configuration= "reg")
m.merge_parameters(model="rf")
m.merge_parameters(model="xgb")
m.merge_parameters(model="knn")
m.total()
m.integrated_totals(targets, depth_w =depth_w, 
                    conversion=conversion,
                    model="abundance_ci50")

#m.merge_env(X_predict)
m.export_ds(current_date + "_abundance_ci50")
m.export_csv(current_date + "_abundance_ci50")


m.estimate_carbon("pg poc")
m.total()
m.export_ds(current_date + "_POC_ci50")
m.export_csv(current_date + "_POC_ci50")
m.integrated_totals(targets, depth_w =depth_w, 
                    conversion=conversion,
                    model="POC_ci50")

m = post(model_config)
#m.merge_env(X_predict)
m.estimate_carbon("pg pic")
m.total()
m.export_ds(current_date + "_PIC_ci50")
m.export_csv(current_date + "_PIC_ci50")
m.integrated_totals(targets, depth_w =depth_w, 
                    conversion=conversion,
                    model="PIC_ci50")





# m = post(model_config, ci=32)
# m.total()
# #m.merge_env(X_predict)
# m.export_ds(current_date + "_abundance_ci32")
# m.export_csv(current_date + "_abundance_ci32")
# m.integrated_totals(targets, depth_w =depth_w, 
#                     conversion=conversion,
#                     model="abundance_ci32")

# m.estimate_carbon("pg poc")
# m.export_ds(current_date + "_POC_ci32")
# m.export_csv(current_date + "_POC_ci32")
# m.integrated_totals(targets, depth_w =depth_w, 
#                     conversion=conversion,
#                     model="POC_ci32")

# m = post(model_config, ci=32)
# m.total()
# #m.merge_env(X_predict)
# m.estimate_carbon("pg pic")
# m.export_ds(current_date + "_PIC_ci32")
# m.export_csv(current_date + "_PIC_ci32")
# m.integrated_totals(targets, depth_w =depth_w, 
#                     conversion=conversion,
#                     model="PIC_ci32")




# m = post(model_config, ci=68)
# m.total()
# #m.merge_env(X_predict)

# m.export_ds(current_date + "_abundance_ci68")
# m.export_csv(current_date + "_abundance_ci68")
# m.integrated_totals(targets, depth_w =depth_w, 
#                     conversion=conversion,
#                     model="abundance_ci68")


# m.estimate_carbon("pg poc")
# m.export_ds(current_date + "_POC_ci68")
# m.export_csv(current_date + "_POC_ci68")
# m.integrated_totals(targets, depth_w =depth_w, 
#                     conversion=conversion,
#                     model="POC_ci68")

# m = post(model_config, ci=68)
# m.total()
# #m.merge_env(X_predict)
# m.estimate_carbon("pg pic")
# m.export_ds(current_date + "_PIC_ci68")
# m.export_csv(current_date + "_PIC_ci68")
# m.integrated_totals(targets, depth_w =depth_w, 
#                     conversion=conversion,
#                     model="PIC_ci68")
