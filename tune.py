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



class tune:

    def __init__(self, X, y, seed, n_threads, verbose, cv, path_out, scale=True):

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


    def tau_scoring(self, y_pred):
        tau, p_value = kendalltau(self.y, y_pred)
        return(tau)

    def tau_scoring_p(self, y_pred):
        tau, p_value = kendalltau(self.y, y_pred)
        return(p_value)
    

        
    class ZeroInflatedRegressor(BaseEstimator, RegressorMixin):
        """
        This was cloned from scikit-lego (0.6.14)
        https://github.com/koaning/scikit-lego
        """

        def __init__(self, classifier, regressor) -> None:
            """Initialize."""
            self.classifier = classifier
            self.regressor = regressor

        def fit(self, X, y, sample_weight=None):
            """
            Fit the model.

            Parameters
            ----------
            X : np.ndarray of shape (n_samples, n_features)
                The training data.

            y : np.ndarray, 1-dimensional
                The target values.

            sample_weight : Optional[np.array], default=None
                Individual weights for each sample.

            Returns
            -------
            ZeroInflatedRegressor
                Fitted regressor.

            Raises
            ------
            ValueError
                If `classifier` is not a classifier or `regressor` is not a regressor.
            """
            X, y = check_X_y(X, y)
            self._check_n_features(X, reset=True)
            if not is_classifier(self.classifier):
                raise ValueError(
                    f"`classifier` has to be a classifier. Received instance of {type(self.classifier)} instead.")
            if not is_regressor(self.regressor):
                raise ValueError(f"`regressor` has to be a regressor. Received instance of {type(self.regressor)} instead.")

            try:
                check_is_fitted(self.classifier)
                self.classifier_ = self.classifier
            except NotFittedError:
                self.classifier_ = clone(self.classifier)

                if "sample_weight" in signature(self.classifier_.fit).parameters:
                    self.classifier_.fit(X, y != 0, sample_weight=sample_weight)
                else:
                #    logging.warning("Classifier ignores sample_weight.")
                    self.classifier_.fit(X, y != 0)

            non_zero_indices = np.where(self.classifier_.predict(X) == 1)[0]

            if non_zero_indices.size > 0:
                try:
                    check_is_fitted(self.regressor)
                    self.regressor_ = self.regressor
                except NotFittedError:
                    self.regressor_ = clone(self.regressor)

                    if "sample_weight" in signature(self.regressor_.fit).parameters:
                        self.regressor_.fit(
                            X[non_zero_indices],
                            y[non_zero_indices],
                            sample_weight=sample_weight[non_zero_indices] if sample_weight is not None else None
                        )
                    else:
                    #    logging.warning("Regressor ignores sample_weight.")
                        self.regressor_.fit(
                            X[non_zero_indices],
                            y[non_zero_indices],
                        )
            else:
                None

            return self


        def predict(self, X):
            """
            Get predictions.

            Parameters
            ----------
            X : np.ndarray, shape (n_samples, n_features)
                Samples to get predictions of.

            Returns
            -------
            y : np.ndarray, shape (n_samples,)
                The predicted values.
            """
            check_is_fitted(self)
            X = check_array(X)
            self._check_n_features(X, reset=False)

            output = np.zeros(len(X))
            non_zero_indices = np.where(self.classifier_.predict(X))[0]

            if non_zero_indices.size > 0:
                output[non_zero_indices] = self.regressor_.predict(X[non_zero_indices])

            return output



    class LogGridSearch:
        def __init__(self, m, verbose, cv, param_grid, scoring):
            self.m = m
            self.verbose = verbose
            self.cv = cv
            self.param_grid = param_grid
            self.scoring = scoring

        def do_log(self, x):
            y = np.log(x+1)
            return(y)
        
        def do_exp(self, x):
            y = np.exp(x)-1
            return(y)
        
        def transformed_fit(self, X, y, log):

            if log=="yes":

                model = TransformedTargetRegressor(self.m, func = self.do_log, inverse_func=self.do_exp)
                grid_search = GridSearchCV(model, param_grid = self.param_grid, scoring=self.scoring, refit="MAE",
                                cv = self.cv, verbose = self.verbose, return_train_score=True, error_score=-1e99)
                grid_search.fit(X, y)
            
            elif log=="no":

                model = TransformedTargetRegressor(self.m, func = None, inverse_func=None)
                grid_search = GridSearchCV(model, param_grid = self.param_grid, scoring=self.scoring, refit="MAE",
                                cv = self.cv, verbose = self.verbose, return_train_score=True, error_score=-1e99)
                grid_search.fit(X, y)

            elif log =="both":

                normal_m = TransformedTargetRegressor(self.m, func = None, inverse_func=None)
                grid_search1 = GridSearchCV(normal_m, param_grid = self.param_grid, scoring=self.scoring, refit="MAE",
                                cv = self.cv, verbose = self.verbose, return_train_score=True, error_score=-1e99)
                grid_search1.fit(X, y)

                log_m = TransformedTargetRegressor(self.m, func = self.do_log, inverse_func=self.do_exp)
                grid_search2 = GridSearchCV(log_m, param_grid = self.param_grid, scoring=self.scoring, refit="MAE",
                                cv = self.cv, verbose = self.verbose, return_train_score=True, error_score=-1e99)
                grid_search2.fit(X, y)

                if (grid_search1.best_score_ > grid_search2.best_score_):
                    grid_search = grid_search1
                    best_transformation = "nolog"
            
                elif (grid_search1.best_score_ < grid_search2.best_score_):
                    grid_search = grid_search2
                    best_transformation = "log"
                else:
                    print("same performance for both models")
                    grid_search = grid_search1
                    best_transformation = "nobest"
                grid_search.cv_results_['best_transformation'] = best_transformation

            else: 
                print("defined log invalid, pick from yes, no or both")

            return grid_search

    
    def XGB(self, reg_scoring,  reg_param_grid, cv, model, clf_scoring=None, clf_param_grid=None, zir=True, log="yes", bagging_estimators=None):

        if model =="xgb":
            classifier = XGBClassifier(nthread=1)
            regressor = XGBRegressor(nthread=1)
        elif model=="knn":
            if bagging_estimators ==None:
                print("you forgot to define the number of bagging estimators")
            else:
                classifier = BaggingClassifier(estimator=KNeighborsClassifier(), n_estimators=bagging_estimators)
                regressor = BaggingRegressor(estimator=KNeighborsRegressor(), n_estimators=bagging_estimators)
        elif model=="rf":
            classifier = RandomForestClassifier(random_state=self.seed, oob_score=True)
            regressor = RandomForestRegressor(random_state=self.seed, oob_score=True)
        else:
            print("invalid model")

        st = time.time()

        if zir==False:
            with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
                reg = self.LogGridSearch(regressor, verbose = self.verbose, cv=cv, param_grid=reg_param_grid, scoring="neg_mean_absolute_error")
                reg_grid_search = reg.transformed_fit(self.X, self.y, log)

            m2 = reg_grid_search.best_estimator_

            pickle.dump(m2, open(self.path_out + self.species + '_reg.sav', 'wb'))

            with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
                reg_scores = cross_validate(m2, self.X, self.y, cv = self.cv, verbose = self.verbose, scoring=reg_scoring)

            reg_sav_out = self.path_out + "scoring/" + model + "/" 

            try: #make new dir if needed
                os.makedirs(reg_sav_out)
            except:
                None

            pickle.dump(reg_scores, open(reg_sav_out + self.species + '_reg.sav', 'wb'))


            print("finished tuning model")
            print("reg rRMSE: " + str(np.mean(reg_scores['test_RMSE'])/np.mean(self.y)))
            print("reg rMAE: " + str(np.mean(reg_scores['test_RMSE'])/np.mean(self.y)))
            print("reg R2: " + str(np.mean(reg_scores['test_R2'])))

        elif zir==True:

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
                reg = self.LogGridSearch(regressor, self.verbose, cv=cv, param_grid=reg_param_grid, scoring="neg_mean_absolute_error")
                reg_grid_search = reg.transformed_fit(self.X, self.y, log)
            m2 = reg_grid_search.best_estimator_


            with parallel_backend('multiprocessing', self.n_jobs):
                clf.fit(self.X, y_clf)

            m1 = clf.best_estimator_

            zir = self.ZeroInflatedRegressor(
                classifier=m1,
                regressor=m2,
            )

            pickle.dump(m1, open(self.path_out + self.species + model + '_clf.sav', 'wb'))
            pickle.dump(m2, open(self.path_out + self.species + model + '_reg.sav', 'wb'))
            pickle.dump(zir, open(self.path_out + self.species + model + '_zir.sav', 'wb'))

            with parallel_backend('multiprocessing', n_jobs=self.n_jobs):
                clf_scores = cross_validate(m1, self.X, y_clf, cv=cv, verbose =self.verbose, scoring=clf_scoring)
                reg_scores = cross_validate(m2, self.X, self.y, cv=cv, verbose =self.verbose, scoring=reg_scoring)
                zir_scores = cross_validate(zir, self.X, self.y, cv=cv, verbose =self.verbose, scoring=reg_scoring)


            zir_sav_out = self.path_out + "scoring/" + model + "/" 

            try: #make new dir if needed
                os.makedirs(zir_sav_out)
            except:
                None

            pickle.dump(clf_scores, open(zir_sav_out + self.species + '_clf.sav', 'wb'))
            pickle.dump(reg_scores, open(zir_sav_out + self.species + '_reg.sav', 'wb'))
            pickle.dump(zir_scores, open(zir_sav_out + self.species + '_zir.sav', 'wb'))

            print("finished tuning model")




            print("reg rRMSE: " + str(np.mean(reg_scores['test_RMSE'])/np.mean(self.y)))
            print("reg rMAE: " + str(np.mean(reg_scores['test_RMSE'])/np.mean(self.y)))
            print("reg R2: " + str(np.mean(reg_scores['test_R2'])))


            print("zir rRMSE: " + str(np.mean(zir_scores['test_RMSE'])/np.mean(self.y)))
            print("zir rMAE: " + str(np.mean(zir_scores['test_RMSE'])/np.mean(self.y)))
            print("zir R2: " + str(np.mean(zir_scores['test_R2'])))

        else: 
            print("zir definition not valid, should be True or False")

        et = time.time()
        elapsed_time = et-st

        print("execution time:", elapsed_time, "seconds")        


    #predict model


    #predict ensemble model

    #out_path = path + "HXGB/parameters/"

        #try: #make new dir if needed
        #    os.makedirs(self.path_out)
        #except:
        #    print("dir already exists")




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