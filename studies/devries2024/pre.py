import pandas as pd
import numpy as np
import glob, os
import xarray as xr
import pickle
import sys

sys.path.insert(0, '/home/phyto/Abil/abil/')

from functions import merge_obs_env

# merge_obs_env(obs_path = "/home/phyto/CASCADE/data/output/gridded_observations.csv",
#                   env_path= '/home/phyto/Abil/devries2024/data/env_data.nc',
#                   env_vars = ["temperature", "si", 
#                               "phosphate", "din", 
#                               "o2", "mld", "DIC", 
#                               "TA", "irradiance", 
#                               "chlor_a", "pic",
#                               "time", "depth", 
#                               "lat", "lon"],
#                     out_path = "/home/phyto/Abil/devries2024/data/obs_env.csv")


d = pd.read_csv("/home/phyto/CASCADE/data/output/cellular_dataset.csv")
df = d.groupby(['species', 'variable'])['value'].mean().reset_index()
df = df.pivot_table(index='species', columns='variable', values='value').reset_index()
df.to_csv("/home/phyto/Abil/devries2024/data/traits.csv")
print("fin")