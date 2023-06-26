import pandas as pd
import pickle
import numpy as np
from skbio.diversity import alpha_diversity
import glob, os
import xarray as xr

class post:

    def __init__(self, path_in, path_out):

        def merge_netcdf(path_in, path_out):
            print("merging...")

            ds = xr.merge([xr.open_dataset(f) for f in glob.glob(os.path.join(path_in, "*.nc"))])
            try: #make new dir if needed
                os.makedirs(path_out)
            except:
                None

            return(ds)
        
        self.path_out = path_out
        self.ds = merge_netcdf(path_in, path_out)
        self.d = self.ds.to_dataframe()
        self.d = self.d.dropna()
        self.species = self.d.columns.values

    def cwm(self, traits, variable):
        w = traits.query('species in @self.species')
        var = w[variable].to_numpy()
        self.d['cwm'] = self.d.apply(lambda row : np.average(var, weights=row[self.species]), axis = 1)
        print("finished calculating CWM " + variable)

    def richness(self, metric):
        measure = alpha_diversity(metric, self.d[self.species])
        self.d[metric] = measure.values
        print("finished calculating " + metric)

    def total(self):
        self.d['total'] = self.d[self.species].sum( axis='columns')
        print("finished calculating total")

    def export_ds(self):
        print(self.d.head())
        self.ds.to_netcdf(self.path_out + "merged.nc")
        print("exported ds to: " + self.path_out + "merged.nc")


