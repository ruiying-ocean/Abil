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
from scipy.stats import kendalltau
from ZeroInflatedRegressor import ZeroInflatedRegressor
from ZeroStratifiedKFold import ZeroStratifiedKFold
from scoring import tau_scoring, tau_scoring_p, tss_scoring
from logGridSearchCV import  LogGridSearch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import time
from sklearn.ensemble import VotingRegressor
import csv
from sklearn.preprocessing import OneHotEncoder

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

    def __init__(self, X, y, envdata, model_config, seed, n_threads, verbose, cv, path_out, scale=True):

        self.y = y

        if scale==True:
            scaler = StandardScaler()  
            scaler.fit(X)  
            X = pd.DataFrame(scaler.transform(X))

        self.X = X
        self.seed = seed
        self.species = "species" #fix so it is the name of y
        self.n_jobs = n_threads
        self.verbose = verbose
        self.path_out = path_out
        self.cv = cv
        self.envdata = envdata
        self.envdata.set_index(['lat', 'lon', 'depth', 'time'], inplace=True)
        self.model_config = model_config
        self.model = self.model_config[0]["config"]

    def do_log(self, x):
        y = np.log(x+1)
        return(y)
    def do_exp(self, x):
        y = np.exp(x)-1
        return(y)


    def calculate_weights(self, mae_dict):

        m1 = -1*mae_dict[0]
        m3 = -1*mae_dict[1]
        m2 = -1*mae_dict[2]

        w =  -1*((m1 +m2+m3)/m)/(((m1 +m2+m3)/m1) + ((m1 +m2+m3)/m2) + ((m1 +m2+m3)/m3))
        return w


    def def_prediction(self, n):
        
        path_to_param = self.model_config[n]['RF_path'] + "parameters/"
        path_to_scores = self.model_config[n]['RF_path'] + "parameters/"

        if self.model_config[n]['config'] == "zir":
            regr_zir_scores = pd.read_csv(path_to_scores + self.species + '.csv')
            regr_zir_mae = regr_zir_scores['zir_MAE'][0]
            regr_reg_mae = regr_zir_scores['reg_MAE'][0]
            if (regr_zir_mae > regr_reg_mae):
                m = pickle.load(open(path_to_param + self.species + '_zir.sav', 'rb'))
                mae = regr_zir_mae
            elif (regr_zir_mae < regr_reg_mae):
                m = pickle.load(open(path_to_param + self.species + '_reg.sav', 'rb'))
                mae = regr_reg_mae
            else:
                m = pickle.load(open(path_to_param + self.species + '_reg.sav', 'rb'))
                mae = regr_reg_mae
        elif self.model_config[n]['config'] == "zir":
            self.m = pickle.load(open(path_to_param + self.species + '_reg.sav', 'rb'))
            regr_zir_scores = pd.read_csv(path_to_scores + self.species + '.csv')
            mae = regr_zir_scores['reg_MAE'][0]
        else:
            print("model config invalid, should be zir or reg")
        return(m, mae)


    def make_prediction(self):

        st = time.time()

        if len(self.model_config)==1:
            m, mae = self.def_prediction(0)
            prediction = m.fit(self.X_train_scaled, self.y).predict(self.X_scaled)
            d = prediction.to_xarray()
            prediction.to_netcdf(self.path_out +"predictions/RF/" + self.species + "_HRF.nc") #remoce spacing from species

        elif len(self.model_config)==2:
            #iteratively make prediction for each model
            m1, mae1 = self.def_prediction(0)
            m2, mae2 = self.def_prediction(1)

            mae_dict = {mae1, mae2}
            w = self.calculate_weights(mae_dict)

            models = list()
            models.append(('knn', m1))  #replace knn with name from dict
            models.append(('rf', m2))
            m = VotingRegressor(estimators=models, weights=w)

            d = m.fit(self.X_train_scaled, self.y).predict(self.X_scaled)
            d.to_netcdf(self.path_out +"predictions/RF/" + self.species + "_HRF.nc") #remoce spacing from species

        et = time.time()
        elapsed_time = et-st
        print("finished")
        print("execution time:", elapsed_time, "seconds")


