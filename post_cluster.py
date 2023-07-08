import sys
import pandas as pd
from yaml import load, Loader
from post import post

with open('/user/work/ba18321/CoccoRandomForestBP/model_config.yml', 'r') as f:
    model_config = load(f, Loader=Loader)

model_config['remote'] = True



with open('/user/work/ba18321/CoccoRandomForestBP/size_groups.yml', 'r') as f:
    size_groups = load(f, Loader=Loader)

dict = {species:k
    for k, v in size_groups.items()
    for species in v['species']}


#load model:
m = post(model_config)

#apply calculations
m.total()
m.cwm("cell diameter")
m.richness('observed_otus')
m.richness('simpson')   
#define groups:
m.def_groups(dict)
#export 
m.export_ds(file_name = "abundances")



#(re)load model:
m = post(model_config)
# convert to carbon:
m.estimate_carbon("POC")

#apply calculations:
m.total()
m.cwm("cell diameter")
m.richness('observed_otus')
m.richness('simpson')

#define groups:
m.def_groups(dict)
#export:
m.export_ds(file_name = "biomass_menden")