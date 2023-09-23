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
from sklearn.ensemble import VotingRegressor, VotingClassifier
import csv
from sklearn.preprocessing import OneHotEncoder
from planktonsdm.functions import do_log, do_exp,  ZeroInflatedRegressor, LogGridSearch, ZeroStratifiedKFold,  UpsampledZeroStratifiedKFold, tau_scoring, tau_scoring_p
from numpy.random import rand

class predict:

    def __init__(self, X_train, y, X_predict, model_config):

        """
        Prediction function

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


    def calculate_weights(self, mae_list, mae):

        m_len = len(mae_list)

        if m_len >= 2:
            mae_sum = sum([-1 * mae for mae in mae_list])
        #    print(mae_sum)
            mae_sums_inverse = sum([(mae_sum / mae) for mae in mae_list])
        #    print(mae_sums_inverse)
            w = -1 * mae_sum / mae_sums_inverse
        else:
            raise ValueError("mae_dict should contain at least 2 elements")

        return w
    

    def calculate_weights(self, n, mae_list):
        m_values = [-1 * mae for mae in mae_list]
        m = mae_list[n]
        
        mae_sum_m = sum(m_values)/m

        mae_sum = sum(m_values)

        mae_sums_inverse = sum([(mae_sum / mae) for mae in mae_list])
        w = mae_sum_m / mae_sums_inverse
        return w
    



    def def_prediction(self, n):

        path_to_scores  = self.path_out + self.ensemble_config["m"+str(n+1)] + "/scoring/"
        path_to_param  = self.path_out +  self.ensemble_config["m"+str(n+1)] + "/model/"


        if (self.ensemble_config["classifier"] ==True) and (self.ensemble_config["regressor"] == False):
            print("predicting classifier")
            m = pickle.load(open(path_to_param + self.species + '_clf.sav', 'rb'))
            scoring =  pickle.load(open(path_to_scores + self.species + '_clf.sav', 'rb'))    
            scores = 1-np.mean(scoring['test_accuracy']) #subtract 1 since lower is better

        elif (self.ensemble_config["classifier"] ==False) and (self.ensemble_config["regressor"] == True):
            print("predicting regressor")
            m = pickle.load(open(path_to_param + self.species + '_reg.sav', 'rb'))
            scoring =  pickle.load(open(path_to_scores + self.species + '_reg.sav', 'rb'))   
            scores = np.mean(scoring['test_MAE'])
 

        elif (self.ensemble_config["classifier"] ==True) and (self.ensemble_config["regressor"] == True):
            print("predicting zero-inflated regressor")
            m = pickle.load(open(path_to_param + self.species + '_zir.sav', 'rb'))
            scoring =  pickle.load(open(path_to_scores + self.species + '_zir.sav', 'rb'))    
            scores = np.mean(scoring['test_MAE'])


        elif (self.ensemble_config["classifier"] ==False) and (self.ensemble_config["regressor"] == False):

            print("Both regressor and classifier are defined as false")



        return(m, scores)
    
    def export_prediction(self, m, ens_model_out):

        d = self.X_predict.copy()
        if (self.model_config['predict_probability'] == True) and (self.ensemble_config["regressor"] ==False):
            print("predicting probabilities")
            d[self.species] = m.predict_proba(self.X_predict)[:, 1]
        elif (self.model_config['predict_probability'] == True) and (self.ensemble_config["regressor"] ==True):
            print("error: can't predict probabilities if the model is a regressor")
        else:
            d[self.species] = m.predict(self.X_predict)
        d = d.to_xarray()
        
        try: #make new dir if needed
            os.makedirs(ens_model_out)
        except:
            None

        d[self.species].to_netcdf(ens_model_out + self.species + ".nc") 



    def score_model(self, m):
        scores = cross_validate(m, self.X_train, self.y, cv=self.cv, verbose =self.verbose, scoring=self.scoring)
        return(scores)



    def make_prediction(self):

        number_of_models = len(self.ensemble_config) -2
        print("number of models in ensemble:" + str(number_of_models))

        if number_of_models==1:

            m, mae1 = self.def_prediction(0)
            model_out = self.path_out + self.ensemble_config["m"+str(1)]["model"] + "/predictions/"
            self.export_prediction(m, model_out)

        elif number_of_models >=2:
                    
            # iteratively make prediction for each model
            models = []
            mae_list = []
            w = []

            for i in range(number_of_models):
                m, mae = self.def_prediction(i)
                model_name = self.ensemble_config["m" + str(i + 1)]
                model_out = self.path_out + model_name + "/predictions/"
                self.export_prediction(m, model_out)
                print("exporting " + model_name + " prediction to: " + model_out)

                models.append((model_name, m))
                mae_list.append(mae)

        
            w = [self.calculate_weights(i, mae_list) for i in range(len(mae_list))]

            if self.ensemble_config["regressor"] ==True:
                m = VotingRegressor(estimators=models, weights=w).fit(self.X_train, self.y)
            else:
                m= VotingClassifier(estimators=models, weights=w).fit(self.X_train, self.y)
            scores = self.score_model(m)

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
