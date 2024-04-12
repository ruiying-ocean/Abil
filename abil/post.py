import pandas as pd
import numpy as np
import glob, os
import xarray as xr
import pickle

if 'site-packages' in __file__:
    from abil.diversity import diversity
else:
    from diversity import diversity

class post:
    """
    Post processing of SDM
    """
    def __init__(self, model_config, ci=50):

        def merge_netcdf(path_in):
            print("merging...")
            #ds = xr.merge([xr.open_dataset(f) for f in glob.glob(os.path.join(path_in, "*.nc"))])
            ds = xr.open_mfdataset(os.path.join(path_in, "*.nc"))
            print("finished loading netcdf files")
            return(ds)
        
        if model_config['hpc']==False:
            self.path_out = model_config['local_root'] + model_config['path_out']  + str(ci) + "/"
            self.ds = merge_netcdf(model_config['local_root'] + model_config['path_in'] + str(ci) + "/")
            self.traits = pd.read_csv(model_config['local_root'] + model_config['targets'])
            self.env_data_path =  model_config['local_root'] + model_config['env_data_path']
            self.root  =  model_config['local_root'] 

        elif model_config['hpc']==True:
            self.path_out = model_config['hpc_root'] + model_config['path_out']  + str(ci) + "/"
            self.ds = merge_netcdf(model_config['hpc_root'] + model_config['path_in'] + str(ci) + "/")
            self.traits = pd.read_csv(model_config['hpc_root'] + model_config['targets'])
            self.env_data_path =  model_config['hpc_root'] + model_config['env_data_path']
            self.root  =  model_config['hpc_root'] 

        else:
            raise ValueError("hpc True or False not defined in yml")
            

        self.d = self.ds.to_dataframe()
        self.d = self.d.dropna()
        self.ds = None
        self.targets = self.d.columns.values
        self.model_config = model_config

    def merge_performance(self, model, configuration=None):
        
        all_performance = []

        if model=="ens":
            extension = ".sav"
        else:
            extension = "_" + configuration + ".sav"

        for i in range(len(self.d.columns)):
            
            m = pickle.load(open(self.root + self.model_config['path_out'] + model + "/scoring/" + self.d.columns[i] + extension, 'rb'))
            mean = np.mean(self.d[self.d.columns[i]])
            R2 = np.mean(m['test_R2'])
            RMSE = -1*np.mean(m['test_RMSE'])
            MAE = -1*np.mean(m['test_MAE'])
            rRMSE = -1*np.mean(m['test_RMSE'])/mean
            rMAE = -1*np.mean(m['test_MAE'])/mean            
            species = self.d.columns[i]
            performance = pd.DataFrame({'species':[species], 'R2':[R2], 'RMSE':[RMSE], 'MAE':[MAE],
                                        'rRMSE':[rRMSE], 'rMAE':[rMAE]})
            all_performance.append(performance)

        all_performance = pd.concat(all_performance)

        if configuration==None:
            all_performance.to_csv(self.root + self.model_config['path_out'] + model + "_performance.csv", index=False)
        else:
            all_performance.to_csv(self.root + self.model_config['path_out'] + model + "_" + configuration + "_performance.csv", index=False)
        
        print("finished merging performance")

    def merge_parameters(self, model):
        
        all_parameters = []

        for i in range(len(self.d.columns)):
            
            species = self.d.columns[i]

            m = pickle.load(open(self.root + self.model_config['path_out'] + model + "/model/" + self.d.columns[i] + "_reg.sav", 'rb'))
            # score_reg = np.mean(m['test_MAE'])

            # m = pickle.load(open(self.root + self.model_config['path_out'] + model + "/scoring/" + self.d.columns[i] + "_zir.sav", 'rb'))
            # score_zir = np.mean(m['test_MAE'])

            # if score_reg > score_zir:
            #     m = pickle.load(open(self.root + self.model_config['path_out'] + model + "/model/" + self.d.columns[i] + "_reg.sav", 'rb'))
            # elif score_reg < score_zir:
            #     m = pickle.load(open(self.root + self.model_config['path_out'] + model + "/model/" + self.d.columns[i] + "_reg.sav", 'rb'))

            if model == "rf":
                max_depth = m.regressor_.named_steps.estimator.max_depth
                max_features = m.regressor_.named_steps.estimator.max_features
                max_samples = m.regressor_.named_steps.estimator.max_samples
                min_samples_leaf = m.regressor_.named_steps.estimator.min_samples_leaf
                parameters = pd.DataFrame({'species':[species], 'max_depth':[max_depth], 'max_features':[max_features], 
                                           'max_samples':[max_samples], 'min_samples_leaf':[min_samples_leaf]})
                all_parameters.append(parameters)
            elif model == "xgb":
                max_depth = m.regressor_.named_steps.estimator.max_depth
                subsample = m.regressor_.named_steps.estimator.subsample
                colsample_bytree = m.regressor_.named_steps.estimator.colsample_bytree

                learning_rate = m.regressor_.named_steps.estimator.learning_rate
                alpha = m.regressor_.named_steps.estimator.reg_alpha

                parameters = pd.DataFrame({'species':[species], 'max_depth':[max_depth], 'subsample':[subsample], 'colsample_bytree':[colsample_bytree],
                                           'learning_rate':[learning_rate], 'alpha':[alpha]                                           
                                           })
                all_parameters.append(parameters)
            elif model == "knn":
                max_features = m.regressor_.named_steps.estimator.max_features
                max_samples = m.regressor_.named_steps.estimator.max_samples

                leaf_size = m.regressor_.named_steps.estimator.estimator.leaf_size
                n_neighbors = m.regressor_.named_steps.estimator.estimator.n_neighbors
                p = m.regressor_.named_steps.estimator.estimator.p
                weights = m.regressor_.named_steps.estimator.estimator.weights

                parameters = pd.DataFrame({'species':[species], 'max_features':[max_features], 'max_samples':[max_samples],
                                           'leaf_size':[leaf_size], 'p':[p], 'n_neighbors':[n_neighbors], 'weights':[weights]
                                           })
                all_parameters.append(parameters)    

        all_parameters= pd.concat(all_parameters)
        all_parameters.to_csv(self.root + self.model_config['path_out'] + model + "_parameters.csv", index=False)
        
        print("finished merging parameters")



    def estimate_carbon(self, variable):

        """
        Estimate carbon content for each species


        Parameters
        ----------

        variable: string
            carbon content to estimate

        """


        w = self.traits.query('Target in @self.targets')
        var = w[variable].to_numpy()
        print(var)
        self.d = self.d.apply(lambda row : (row[self.targets]* var), axis = 1)
        print("finished estimating " + variable)


    def def_groups(self, dict):
        """
        Define groups of species

        Parameters
        ----------

        dict: dictionary
        A dictionary containing group definitions

        """     

        df = self.d[self.targets]
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

        w = self.traits.query('Target in @self.targets')
        var = w[variable].to_numpy()
        var_name = 'cwm ' + variable
        self.d[var_name] = self.d.apply(lambda row : np.average(var, weights=row[self.targets]), axis = 1)
        print("finished calculating CWM " + variable)

    def richness(self, metric):
        measure = diversity(metric, self.d[self.targets].clip(lower=1))
        self.d[metric] = measure.values
        print("finished calculating " + metric)

    def total(self):
        """
        Sum target rows to estimate total.

        Notes
        ----------
        Useful for estimating total species abundances if targets are continuous.
        Total is estimated based on the target list defined in model_config. 

        """

        self.d['total'] = self.d[self.targets].sum( axis='columns')
        self.d['total_log'] = np.log(self.d['total'])
        print("finished calculating total")


    # work in progress:
    def integrated_total(self, variable="total", lat_name="lat", 
                         depth_w =5, conversion=1e3):
        """
        Estimates global integrated values for a single target.

        Considers latitude and depth bin size.

        depth_w should be in meters

        conversion should be in cubic meters. 
        e.g. if original unit is cells/l conversion should be = 1e3

        """
        def asRadians(degrees):
            return degrees * np.pi / 180

        df = self.d[self.d[variable]>0].reset_index()[[lat_name, variable]].copy()
        if lat_name not in df:
            raise ValueError("lat_name not defined in dataframe")

        #convert lat and lon to meters:
        lat_w = (40075000 * np.cos(asRadians(df[lat_name]))) / 360
        lon_w = 11132

        total = np.sum(df[variable]*lat_w*depth_w*lon_w*conversion)

        return(total)

    def integrated_totals(self, targets, lat_name="lat", 
                         depth_w =5, conversion=1e3, 
                         export=True, model="ens"):
        """
        Estimates global integrated values for all targets.

        Considers latitude and depth bin size.

        depth_w should be in meters

        conversion should be in cubic meters. 
        e.g. if original unit is cells/l conversion should be = 1e3

        """

        if 'total' in self.d:
            targets = np.append(targets, 'total')

        totals = []

        for i in range(0, len(targets)):
            target = targets[i]
            
            try:
                total = pd.DataFrame({'total': [self.integrated_total(variable=target, lat_name=lat_name, 
                                depth_w =depth_w, conversion=conversion)], 'variable':target
                                })
                
                totals.append(total)
            except:
                print("some targets do not have predictions!")
                print("missing: " + target)

        totals = pd.concat(totals, ignore_index=True)

        if export:
            totals.to_csv(self.root + self.model_config['path_out'] + model + "_integrated_totals.csv", index=False)
            print("exported totals to: " + self.root + self.model_config['path_out'] + model + "_integrated_totals.csv")
        



    def merge_env(self, X_predict):
        """
        Merge model output with environmental data 
        """

        self.d = pd.concat([self.d, X_predict], axis=1)

    def return_d(self):
        return(self.d)

    def export_ds(self, file_name, 
                  author=None, description=None):
        """
        Export processed dataset to netcdf.

        Parameters
        ----------
        file_name: name netcdf will be saved as. 
        author: author included in netcdf description
        description: title included in netcdf description

        Notes
        ----------
        data export location is defined in the model_config.yml

        """
    
        try: #make new dir if needed
            os.makedirs(self.path_out)
        except:
            None

        print("export_ds")
        print("dataframe: ")
        print(self.d.head())
        ds = self.d.to_xarray()

        if description is not None:
            ds.attrs['description'] = description
        ds.attrs['Conventions'] = 'CF-1.5'
        if author is not None:
            ds.attrs['creator_name'] = author

        ds['lat'].attrs['units'] = 'degrees_north'
        ds['lat'].attrs['long_name'] = 'latitude'

        ds['lon'].attrs['units'] = 'degrees_east'
        ds['lon'].attrs['long_name'] = 'longitude'

        ds['depth'].attrs['units'] = 'm'
        ds['depth'].attrs['positive'] = 'down'

        #to add loop defining units of variables

        print(self.d.head())
        ds.to_netcdf(self.path_out + file_name + ".nc")
        print("exported ds to: " + self.path_out + file_name + ".nc")
        #add nice metadata


    def export_csv(self, file_name):
        """
        Export processed dataset to csv.

        Parameters
        ----------
        file_name: name csv will be saved as. 

        Notes
        ----------
        data export location is defined in the model_config.yml

        """
    
        try: #make new dir if needed
            os.makedirs(self.path_out)
        except:
            None
    
        print(self.d.head())
        self.d.to_csv(self.path_out + file_name + ".csv")
        print("exported d to: " + self.path_out + file_name + ".csv")
        #add nice metadata
