from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.compose import TransformedTargetRegressor
import pandas as pd
import numpy as np
import pickle
import sys
from joblib import parallel_backend
import multiprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, f1_score, precision_score, recall_score, balanced_accuracy_score
import random
import math
import os
import datetime
from sklearn.metrics import make_scorer
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import time
from sklearn.ensemble import VotingRegressor
import csv
from sklearn.preprocessing import OneHotEncoder
from functions import do_log, do_exp,  ZeroInflatedRegressor, LogGridSearch, ZeroStratifiedKFold, tau_scoring, tau_scoring_p
from numpy.random import rand

'''

model_config = {
    "RF": {
        "path":"/home/phyto/CoccoML/ModelOutput",
        "config": "zir"
    },
    "XGB": {
        "path":"/home/phyto/CoccoML/ModelOutput",
        "config": "zir"
    },
    "KNN": {
        "path":"/home/phyto/CoccoML/ModelOutput",
        "config": "zir"
    }
}

#for each model in dictionary make prediction
if n>2 and ensemble = True then make an ensemble model weighted based on scores

'''

class predict:

    def __init__(self, X, y, envdata, model_config):


        if model_config['scale_X']==True:
            scaler = StandardScaler()  
            scaler.fit(X)  
            X = pd.DataFrame(scaler.transform(X))

        self.X = X
        self.seed = model_config['seed']
        self.species = y.columns[0] 
        self.n_jobs = model_config['n_threads']
        self.verbose = model_config['verbose']
        self.path_out =model_config['path_out']
        self.cv = model_config['cv']
        self.envdata = envdata
        self.model_config = model_config
        self.y = y[self.species].ravel()

    def calculate_weights(self, m, mae_dict):

        m1 = -1*mae_dict[0]
        m3 = -1*mae_dict[1]
        m2 = -1*mae_dict[2]

        w =  -1*((m1 +m2+m3)/m)/(((m1 +m2+m3)/m1) + ((m1 +m2+m3)/m2) + ((m1 +m2+m3)/m3))
        return w

    def def_prediction(self, n):
        
        path_to_param = self.model_config[list(self.model_config)[n]]['path'] + "model/"
        path_to_scores = self.model_config[list(self.model_config)[n]]['path'] + "scoring/"
        if self.model_config[list(self.model_config)[n]]['config'] == "zir":
            regr_zir_scores = pickle.load(open(path_to_scores + self.species + '_zir.sav', 'rb'))
            regr_reg_scores = pickle.load(open(path_to_scores + self.species + '_reg.sav', 'rb'))

            regr_zir_mae = np.mean(regr_zir_scores['test_MAE'])
            regr_reg_mae = np.mean(regr_reg_scores['test_MAE'])

            if (regr_zir_mae > regr_reg_mae):
                m = pickle.load(open(path_to_param + self.species + '_zir.sav', 'rb'))
                mae = regr_zir_mae
            elif (regr_zir_mae < regr_reg_mae):
                m = pickle.load(open(path_to_param + self.species + '_reg.sav', 'rb'))
                mae = regr_reg_mae
            else:
                m = pickle.load(open(path_to_param + self.species + '_reg.sav', 'rb'))
                mae = regr_reg_mae
        elif self.model_config[list(self.model_config)[n]]['config'] == "reg":
            m = pickle.load(open(path_to_param + self.species + '_reg.sav', 'rb'))
            regr_reg_scores = pickle.load(open(path_to_scores + self.species + '_reg.sav', 'rb'))            
            mae = np.mean(regr_reg_scores['test_MAE'])

        else:
            print("model config invalid, should be zir or reg")
        return(m, mae)
    
    def export_prediction(self, m, ens_model_out):

        d = self.envdata.copy()
        d[self.species] = m.fit(self.X, self.y).predict(self.envdata)
        d = d.to_xarray()
        
        try: #make new dir if needed
            os.makedirs(ens_model_out)
        except:
            None

        d[self.species].to_netcdf(ens_model_out + self.species + ".nc") #remoce spacing from species


    def make_prediction(self):

        st = time.time()

        if len(self.model_config)==1:

            m, mae1 = self.def_prediction(0)
            model_out = self.path_out + list(self.model_config)[0] + "/predictions/"
            self.export_prediction(m, model_out)

        elif len(self.model_config)==2:
            #iteratively make prediction for each model
            m1, mae1 = self.def_prediction(0)
            model_out = self.path_out + list(self.model_config)[0] + "/predictions/"
            self.export_prediction(m1, model_out)

            m2, mae2 = self.def_prediction(1)
            model_out = self.path_out + list(self.model_config)[1] + "/predictions/"
            self.export_prediction(m2, model_out)

            mae_dict = {mae1, mae2}
            w = self.calculate_weights(mae_dict)

            models = list()
            models.append((list(self.model_config)[0], m1))  #replace knn with name from dict
            models.append((list(self.model_config)[1], m2))
            m = VotingRegressor(estimators=models, weights=w)
            model_out = self.path_out + "ens/predictions/"
            self.export_prediction(m, model_out)

        elif len(self.model_config)==3:

            #iteratively make prediction for each model
            m1, mae1 = self.def_prediction(0)
            model_out = self.path_out + list(self.model_config)[0] + "/predictions/"
            self.export_prediction(m1, model_out)

            m2, mae2 = self.def_prediction(1)
            model_out = self.path_out + list(self.model_config)[1] + "/predictions/"
            self.export_prediction(m2, model_out)

            m3, mae3 = self.def_prediction(2)
            model_out = self.path_out + list(self.model_config)[2] + "/predictions/"
            self.export_prediction(m3, model_out)

            mae_dict = [mae1, mae2, mae3]

            print(mae_dict)
            w1 = self.calculate_weights(mae1, mae_dict)
            w2 = self.calculate_weights(mae2, mae_dict)
            w3 = self.calculate_weights(mae3, mae_dict)

            w = [w1, w2, w3]
            
            models = list()
            models.append((list(self.model_config)[0], m1)) 
            models.append((list(self.model_config)[1], m2))
            models.append((list(self.model_config)[2], m3))
            m = VotingRegressor(estimators=models, weights=w)

            model_out = self.path_out + "ens/predictions/"
            d = self.export_prediction(m, model_out)

        else:
            print("wtf")
        et = time.time()
        elapsed_time = et-st
        print("finished")
        print("execution time:", elapsed_time, "seconds")


'''

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from functions import example_data
from numpy.random import rand
from predict import predict

seed = 1
n_threads = 4
verbose = 1 
cv = 3
path_out = "/home/phyto/ModelOutput/test/"

X, y = example_data(y_name =  "Emiliania huxleyi", n_samples=500, n_features=5, noise=20, random_state=seed)

envdata = pd.DataFrame({"no3": rand(50), "mld": rand(50), "par": rand(50), "o2": rand(50), "temp": rand(50),
                        "lat": range(0,50, 1), "lon": range(0,50, 1)})
envdata.set_index(['lat', 'lon'], inplace=True)


model_config = {
    "rf": {
        "path":"/home/phyto/ModelOutput/test/rf/",
        "config": "zir"
    },
    "xgb": {
        "path":"/home/phyto/ModelOutput/test/xgb/",
        "config": "zir"
    },
    "knn": {
        "path":"/home/phyto/ModelOutput/test/knn/",
        "config": "zir"
    }
}

m = predict(X, y, envdata, model_config, seed, n_threads, verbose, cv, path_out, scale=True)
m.make_prediction()

X, y = example_data(y_name =  "Coccolithus pelagicus", n_samples=500, n_features=5, noise=20, random_state=seed)

m = predict(X, y, envdata, model_config, seed, n_threads, verbose, cv, path_out, scale=True)
m.make_prediction()


print("fin")

'''