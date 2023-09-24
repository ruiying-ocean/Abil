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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import time
from sklearn.ensemble import VotingRegressor, VotingClassifier
import csv
from sklearn.preprocessing import OneHotEncoder
from planktonsdm.functions import calculate_weights, score_model, def_prediction, do_log, do_exp,  ZeroInflatedRegressor, LogGridSearch, ZeroStratifiedKFold,  UpsampledZeroStratifiedKFold, tau_scoring, tau_scoring_p


class predict:
    """
    Parameters
    ----------

    X_train : {array-like, sparse matrix} of shape (n_samples, n_features)
        should be the same as the X used for tuning

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        should be the same as the y used for tuning

    X_predict : {array-like, sparse matrix} of shape (n_samples, n_features)
        Features to predict on (i.e. gridded environmental data). 
            
    model_config: dictionary, default=None
        A dictionary containing:

        `seed` : int, used to create random numbers

        `root`: string, path to folder
        
        `path_out`: string, where predictions are saved
        
        `path_in`: string, where to find tuned models
        
        `traits`: string, file name of your trait file
        
        `verbose`: int, to set verbosity (0-3)
        
        `n_threads`: int, number of threads to use
        
        `cv` : int, number of cross-folds
                    
        `ensemble_config` : 
        
        `clf_scoring` :
        
        `reg_scoring` :

    """
    def __init__(self, X_train, y, X_predict, model_config):

        
        self.st = time.time()


        if model_config['scale_X']==True:
            scaler = StandardScaler()  
            scaler.fit(X_train)  
            X_train = pd.DataFrame(scaler.transform(X_train))
            print("scale X = True")

        self.y = y.sample(frac=1, random_state=model_config['seed']) #shuffle

        self.X_train = X_train
        self.seed = model_config['seed']
        self.species = y.name
        self.n_jobs = model_config['n_threads']
        self.verbose = model_config['verbose']
        self.path_out = model_config['root'] + model_config['path_out']

        if model_config['upsample']==True:
            self.cv = UpsampledZeroStratifiedKFold(n_splits=model_config['cv'])
            print("upsampling = True")

        else:
            self.cv = ZeroStratifiedKFold(n_splits=model_config['cv'])

        if model_config['scale_X']==True:
            scaler = StandardScaler()  
            scaler.fit(X_train)  
            X_predict = pd.DataFrame(scaler.transform(X_predict))
            print("scaled X_predict")

        self.X_predict = X_predict
        self.ensemble_config = model_config['ensemble_config']
        self.model_config = model_config


        if (self.ensemble_config["classifier"] ==True) and (self.ensemble_config["regressor"] == False):
            self.scoring = model_config['clf_scoring']
            self.y[self.y > 0] = 1

        elif (self.ensemble_config["classifier"] ==False) and (self.ensemble_config["regressor"] == False):
            raise ValueError("classifier and regressor can't both be False")
        else:
            self.scoring = model_config['reg_scoring']


        if (self.ensemble_config["classifier"] !=True) and (self.ensemble_config["classifier"] !=False):
            raise ValueError("classifier should be True or False")
        
        if (self.ensemble_config["regressor"] !=True) and (self.ensemble_config["regressor"] !=False):
            raise ValueError("regressor should be True or False")

    def make_prediction(self):


        """
        Calculates performance of model(s) and exports prediction(s) to netcdf

        Notes
        -----
        If more than one model is provided, predictions are made for both invidiual models 
        and an ensemble of the models. 
        """


        number_of_models = len(self.ensemble_config) -2
        print("number of models in ensemble:" + str(number_of_models))

        if number_of_models==1:

            m, mae1 = def_prediction(self.path_out, self.ensemble_config, 0, self.species)

            model_out = self.path_out + self.ensemble_config["m"+str(1)]["model"] + "/predictions/"
            self.export_prediction(m, model_out)

        elif number_of_models >=2:
                    
            # iteratively make prediction for each model
            models = []
            mae_list = []
            w = []

            for i in range(number_of_models):
                m, mae = def_prediction(self.path_out, self.ensemble_config, i, self.species)
                model_name = self.ensemble_config["m" + str(i + 1)]
                model_out = self.path_out + model_name + "/predictions/"
                self.export_prediction(m, model_out)
                print("exporting " + model_name + " prediction to: " + model_out)

                models.append((model_name, m))
                mae_list.append(mae)

        
            w = [calculate_weights(i, mae_list) for i in range(len(mae_list))]

            if self.ensemble_config["regressor"] ==True:
                m = VotingRegressor(estimators=models, weights=w).fit(self.X_train, self.y)
            else:
                m= VotingClassifier(estimators=models, weights=w).fit(self.X_train, self.y)
            scores = score_model(m, self.X_train, self.y, self.cv, self.verbose, self.scoring)


            model_out = self.path_out + "ens/predictions/"
            try: #make new dir if needed
                os.makedirs(model_out)
            except:
                None

            print("predicting ensemble model")

            self.export_prediction(m, model_out)
            print("exporting ensemble prediction to: " + model_out)


            model_out_scores = self.path_out + "ens/scoring/"

            try: #make new dir if needed
                os.makedirs(model_out_scores)
            except:
                None

            pickle.dump(scores, open(model_out_scores + self.species + '.sav', 'wb'))   
            

            print("exporting ensemble scores to: " + model_out_scores)


        et = time.time()
        elapsed_time = et-self.st
        print("finished")
        print("execution time:", elapsed_time, "seconds")



