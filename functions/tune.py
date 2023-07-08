import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ["OMP_THREAD_LIMIT"] = "1"

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pickle
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder
from joblib import parallel_backend
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats import kendalltau
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.base import BaseEstimator, RegressorMixin, clone, is_regressor, is_classifier
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.exceptions import NotFittedError
from inspect import signature
import logging
from functions import do_log, do_exp,  ZeroInflatedRegressor, LogGridSearch, ZeroStratifiedKFold, tau_scoring, tau_scoring_p

class tune:

    def __init__(self, X, y, model_config, scale=True):
        self.y = y
        if scale==True:
            scaler = StandardScaler()  
            scaler.fit(X)  
            self.X = pd.DataFrame(scaler.transform(X))
        else:
            self.X = X
        self.model_config = model_config
        self.seed = model_config['seed']
        self.species = y.columns[0] 
        self.n_jobs = model_config['n_threads']
        self.verbose = model_config['verbose'] 
        self.path_out =  model_config['local_root'] + model_config['path_out']
        self.cv = model_config['cv'] 
        try:
            self.bagging_estimators = model_config['knn_bagging_estimators'] 
        except:
            self.bagging_estimators = None

    
    def train(self, model, zir=False, log="no"):

        reg_scoring = self.model_config['reg_scoring']
        reg_param_grid = self.model_config[model + '_param_grid']['reg_param_grid']

        if model =="xgb":
            classifier = XGBClassifier(nthread=1)
            regressor = XGBRegressor(nthread=1)
        elif model=="knn":
            if self.bagging_estimators ==None:
                print("you forgot to define the number of bagging estimators")
            else:
                classifier = BaggingClassifier(estimator=KNeighborsClassifier(), n_estimators=self.bagging_estimators)
                regressor = BaggingRegressor(estimator=KNeighborsRegressor(), n_estimators=self.bagging_estimators)
        elif model=="rf":
            classifier = RandomForestClassifier(random_state=self.seed, oob_score=True)
            regressor = RandomForestRegressor(random_state=self.seed, oob_score=True)
        else:
            print("invalid model")

        st = time.time()


        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            reg = LogGridSearch(regressor, verbose = self.verbose, cv=self.cv, param_grid=reg_param_grid, scoring="neg_mean_absolute_error")
            reg_grid_search = reg.transformed_fit(self.X, self.y[self.species].ravel(), log)

        m2 = reg_grid_search.best_estimator_

        reg_sav_out = self.path_out + model + "/scoring/"

        try: #make new dir if needed
            os.makedirs(reg_sav_out)
        except:
            None

        pickle.dump(m2, open(reg_sav_out  + self.species + '_reg.sav', 'wb'))

        with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
            reg_scores = cross_validate(m2, self.X, self.y[self.species].ravel(), cv = self.cv, verbose = self.verbose, scoring=reg_scoring)

        pickle.dump(reg_scores, open(reg_sav_out + self.species + '_reg.sav', 'wb'))

        print("finished tuning model")

        print("reg rRMSE: " + str(int(round(np.mean(reg_scores['test_RMSE'])/np.mean(self.y), 2)*-100))+"%")
        print("reg rMAE: " + str(int(round(np.mean(reg_scores['test_MAE'])/np.mean(self.y), 2)*-100))+"%")
        print("reg R2: " + str(round(np.mean(reg_scores['test_R2']), 2)))


        if zir==True:
            clf_param_grid = self.model_config[model + '_param_grid']['clf_param_grid']
            clf_scoring = self.model_config['clf_scoring']

            clf = GridSearchCV(
                estimator=classifier,
                param_grid= clf_param_grid,
                scoring= 'balanced_accuracy',
                cv = self.cv,
                verbose = self.verbose
            )

            y_clf =  self.y.copy()
            y_clf[y_clf > 0] = 1

            with parallel_backend('multiprocessing', self.n_jobs):
                clf.fit(self.X, y_clf[self.species].ravel())

            m1 = clf.best_estimator_

            zir = ZeroInflatedRegressor(
                classifier=m1,
                regressor=m2,
            )

            zir_scores_out = self.path_out + model + "/scoring/" 

            try: #make new dir if needed
                os.makedirs(zir_scores_out)
            except:
                None
            zir_sav_out = self.path_out + model + "/model/"

            try: #make new dir if needed
                os.makedirs(zir_sav_out)
            except:
                None           

            pickle.dump(m1, open(zir_sav_out + self.species + '_clf.sav', 'wb'))
            pickle.dump(zir, open(zir_sav_out + self.species + '_zir.sav', 'wb'))

            with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
                clf_scores = cross_validate(m1, self.X, y_clf[self.species].ravel(), cv=self.cv, verbose =self.verbose, scoring=clf_scoring)
                zir_scores = cross_validate(zir, self.X, self.y[self.species].ravel(), cv=self.cv, verbose =self.verbose, scoring=reg_scoring)

            zir_scores_out = self.path_out + model + "/scoring/" 

            try: #make new dir if needed
                os.makedirs(zir_scores_out)
            except:
                None

            pickle.dump(clf_scores, open(zir_scores_out + self.species + '_clf.sav', 'wb'))
            pickle.dump(zir_scores, open(zir_scores_out + self.species + '_zir.sav', 'wb'))

            print("zir rRMSE: " + str(int(round(np.mean(zir_scores['test_RMSE'])/np.mean(self.y), 2)*-100))+"%")
            print("zir rMAE: " + str(int(round(np.mean(zir_scores['test_MAE'])/np.mean(self.y), 2)*-100))+"%")
            print("zir R2: " + str(round(np.mean(zir_scores['test_R2']), 2)))
        else: 
            None
        

        et = time.time()
        elapsed_time = et-st

        print("execution time:", elapsed_time, "seconds")        





'''
EXAMPLE 1 (local version)

st = time.time()

seed = 1 #random seed
n_threads = 2 # how many cpu threads to use
n_spp = 0 # which species to model
path_out = "/home/phyto/ModelOutput/test/" #where to save model output

d = pd.DataFrame({"mld":[50, 100, 120, 50, 50, 100, 120, 50, 200], 
                    "temperature":[25, 20, 15, 45, 25, 20, 15, 45, 10],
                    "Emiliania huxleyi":[300000, 100000, 0, 6000, 9000, 5000, 3000, 0, 0],
                    "Coccolithus pelagicus":[50000, 30000, 500, 800, 900, 0, 1000, 5000, 0]
})

dict =  {

    "X_vars" : ["mld", "temperature"],

    "species" : ["Emiliania huxleyi", "Coccolithus pelagicus"],

    "reg_scoring" : {
                "R2":"r2",
                "MAE": "neg_mean_absolute_error", 
                "RMSE":"neg_root_mean_squared_error",
                },

    "reg_param_grid" : {
                    'regressor__eta': [0.01],
                    'regressor__n_estimators': [10],
                    'regressor__max_depth': [4],
                    'regressor__colsample_bytree': [0.6],               
                    'regressor__subsample': [0.6],          
                    'regressor__gamma': [1],
                    'regressor__alpha': [1]
                    },


    "clf_scoring" : {
                "accuracy": "balanced_accuracy",
                },

    "clf_param_grid" : {    
                    'eta':[0.01],       
                    'n_estimators': [10],
                    'max_depth': [4],
                    'subsample': [0.6],  
                    'colsample_bytree': [0.6],  
                    'gamma':[1],     
                    'alpha':[1]   
                    },                    

}

X_vars = dict["X_vars"] 
species = dict["species"][n_spp]
reg_scoring = dict['reg_scoring']
reg_param_grid = dict['reg_param_grid']
cv = 3
verbose = 3


#initiate model and load data:
m = tune(d, X_vars, species, seed, n_threads, verbose, cv, path_out,  hot_encode=False)

#run regressor model:
#m.XGB(reg_scoring, reg_param_grid, cv=cv, model="reg", log="yes")

et = time.time()
elapsed_time = et-st
print("elapsed time:" + elapsed_time)

EXAMPLE 2 (cluster version)

seed = 1
n_threads = sys.arg[0]
n_species = sys.arg[1]
path = "cluster_path/data/"
path_out = "cluster_path/out/"

dict =  {

    X_vars = ["temperature", "mld"],

    species = ["Emiliania huxleyi", "Coccolithus pelagicus"],

    clf_scoring = {
                "accuracy": "balanced_accuracy"
                },

    reg_scoring = {
                "R2":"r2",
                "tau": make_scorer(tau_scoring),
                "tau_p": make_scorer(tau_scoring_p, greater_is_better=False),
                "MAE": "neg_mean_absolute_error", 
                "RMSE":"neg_root_mean_squared_error",
                },

                

    
    clf_param_grid = {    
                    'eta': eta,       
                    'n_estimators': n_estimators,  
                    'max_depth': max_depth,  
                    'subsample': sub_sample,   
                    'colsample_bytree': colsample_bytree,
                    'gamma':gamma,       
                    'alpha':alpha    
                    },

    reg_param_grid = {
                    'regressor__eta': eta,
                    'regressor__n_estimators': n_estimators,
                    'regressor__max_depth': max_depth,
                    'regressor__colsample_bytree':colsample_bytree,               
                    'regressor__subsample': sub_sample,          
                    'regressor__gamma': gamma,
                    'regressor__alpha': alpha
                    }

}

X_vars = dict["X_vars"] 
species = dict["species"][n_spp]
d = pd.read_csv(path + "abundances_environment.csv")
clf_scoring = dict['scoring']
clf_param_grid = dict['param_grid']
reg_scoring = dict['reg_scoring']
reg_param_grid = dict['reg_param_grid']
cv = 10

#run zir model:
#m.XGB(reg_scoring, reg_param_grid, clf_scoring = clf_scoring, clf_param_grid = clf_param_grid, cv=cv, model="zir", log="both")

#run zir model:
m.XGB.tune(clf_scoring, reg_scoring, clf_param_grid, reg_param_grid, model="zir", log="both")
m.XGB.export_csv()
m.XGB.export_sav()
X_pred = m.XGB.predict()
X_pred.to_csv(path_out)

run knn model:


predict ensemble model:




'''