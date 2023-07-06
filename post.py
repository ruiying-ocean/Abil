import pandas as pd
import pickle
import numpy as np
from skbio.diversity import alpha_diversity
import glob, os
import xarray as xr

class post:

    def __init__(self, model_config):

        def merge_netcdf(path_in):
            print("merging...")
            ds = xr.merge([xr.open_dataset(f) for f in glob.glob(os.path.join(path_in, "*.nc"))])
            return(ds)
        
        if model_config['remote']==False:
            self.path_out = model_config['local_root'] + model_config['path_out'] 
            self.ds = merge_netcdf("/home/phyto/CoccoML/ModelOutput/test/ens/predictions/" )
            self.traits = pd.read_csv("/home/phyto/CoccoML/data/traits.csv")

        else:
            self.path_out = model_config['remote_root'] + model_config['path_out'] 
            self.ds = merge_netcdf(model_config['remote_root'] + model_config['path_in'] )   
            self.traits = pd.read_csv(model_config['traits'])

        self.d = self.ds.to_dataframe()
        self.d = self.d.dropna()
        self.ds = None
        self.species = self.d.columns.values

    def estimate_carbon(self, variable):
        w = self.traits.query('species in @self.species')
        var = w[variable].to_numpy()
        print(var)
        self.d = self.d.apply(lambda row : (row[self.species]* var), axis = 1)
        print("finished estimating " + variable)

    def cwm(self, variable):
        w = self.traits.query('species in @self.species')
        var = w[variable].to_numpy()
        var_name = 'cwm ' + variable
        self.d[var_name] = self.d.apply(lambda row : np.average(var, weights=row[self.species]), axis = 1)
        print("finished calculating CWM " + variable)

    def richness(self, metric):
        measure = alpha_diversity(metric, self.d[self.species].clip(lower=1))
        self.d[metric] = measure.values
        print("finished calculating " + metric)

    def total(self):
        self.d['total'] = self.d[self.species].sum( axis='columns')
        print("finished calculating total")

    def export_ds(self, file_name):
        try: #make new dir if needed
            os.makedirs(self.path_out)
        except:
            None
    
        ds = self.d.to_xarray()

        print(self.d.head())
        ds.to_netcdf(self.path_out + file_name + ".nc")
        print("exported ds to: " + self.path_out + file_name + ".nc")
        #add nice metadata
