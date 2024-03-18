import pandas as pd
import numpy as np
import glob, os
import xarray as xr
import pickle
from functions import merge_obs_env, longhurst_gridding


d = merge_obs_env(obs_path = "/home/phyto/Abil/data/gridded_observations.csv",
                  env_path= '/home/phyto/Abil/data/env_data.nc',
                  env_vars = ["temperature", "si", 
                              "phosphate", "din", 
                              "o2", "mld", "DIC", 
                              "TA", "irradiance", 
                              "chlor_a","Rrs_547",
                              "Rrs_667", "pic", "FID", 
                              "time", "depth", 
                              "lat", "lon"],
                    out_path = "/home/phyto/Abil/data/obs_env.csv")


print("fin")