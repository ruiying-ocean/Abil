if __name__ == "__main__":
	from skbio.diversity import alpha_diversity
import pandas as pd
import pickle
import numpy as np

import glob, os
import xarray as xr

class post:
    """
    Post processing of SDM
    """
    def __init__(self, model_config):

        def merge_netcdf(path_in):
            print("merging...")
            ds = xr.merge([xr.open_dataset(f) for f in glob.glob(os.path.join(path_in, "*.nc"))])
            return(ds)
        
        if model_config['remote']==False:
            self.path_out = model_config['local_root'] + model_config['path_out'] 
        #    self.ds = merge_netcdf("/home/phyto/CoccoML/ModelOutput/test/ens/predictions/" )
            self.ds = merge_netcdf(model_config['local_root'] + model_config['path_in'] )
            self.traits = pd.read_csv("/home/phyto/CoccoML/data/traits.csv")

        else:
            self.path_out = model_config['remote_root'] + model_config['path_out'] 
            self.ds = merge_netcdf(model_config['remote_root'] + model_config['path_in'] )   
            self.traits = pd.read_csv(model_config['traits'])

        self.d = self.ds.to_dataframe()
        self.d = self.d.dropna()
        self.ds = None
        self.species = self.d.columns.values
        self.env_data_path =  model_config['env_data_path']

    def merge_performance(self):

        
        print("finished merging performance")


    def estimate_carbon(self, variable):

        """
        Estimate carbon content for each species


        Parameters
        ----------

        variable: string
            carbon content to estimate

        """


        w = self.traits.query('species in @self.species')
        var = w[variable].to_numpy()
        print(var)
        self.d = self.d.apply(lambda row : (row[self.species]* var), axis = 1)
        print("finished estimating " + variable)


    def def_groups(self, dict):
        """
        Define groups of species

        Parameters
        ----------

        dict: dictionary
        A dictionary containing group definitions


        """
                

        df = self.d[self.species]
        df = (df.rename(columns=dict)
            .groupby(level=0, axis=1, dropna=False)).sum( min_count=1)
        self.d = pd.concat([self.d, df], axis=1)
        print("finished defining groups")

    def cwm(self, variable):
        """
        Calculate community weighted mean values for a given parameter. 

        Parameters
        ----------

        variable: string
            variable that is used to estimate cwm.

        """

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
        """
        Calculate total

        Notes
        ----------

        Total is estimated based on the species list defined in model_config. Other species or groupings are excluded from the summation. 

        """

        self.d['total'] = self.d[self.species].sum( axis='columns')
        self.d['total_log'] = np.log(self.d['total'])
        print("finished calculating total")

    def merge_env(self):
        """
        Merge model output with environmental data 

        """


        env_data = pd.read_csv(self.env_data_path)
        env_data.set_index(["time", "depth", "lat", "lon"], inplace=True)
        print(self.d.head())
        self.d.reset_index(inplace=True)
        self.d.set_index(["time", "depth", "lat", "lon"], inplace=True)

        self.d = pd.concat([self.d, env_data], axis=1)

    def return_d(self):
        return(self.d)

    def export_ds(self, file_name):
        """
        Export processed dataset to netcdf.

        Parameters
        ----------
        file_name: name netcdf will be saved as. 

        Notes
        ----------
        data export location is defined in the model_config.yml

        """

    
        try: #make new dir if needed
            os.makedirs(self.path_out)
        except:
            None
    
        ds = self.d.to_xarray()

        print(self.d.head())
        ds.to_netcdf(self.path_out + file_name + ".nc")
        print("exported ds to: " + self.path_out + file_name + ".nc")
        #add nice metadata
