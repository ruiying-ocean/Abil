from sklearn.base import BaseEstimator, RegressorMixin, clone, is_regressor, is_classifier
from scipy.stats import kendalltau
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.exceptions import NotFittedError
from inspect import signature
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_validate
import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import xarray as xr
import numpy as np
from sklearn.metrics import make_scorer

def tau_scoring(y, y_pred):
    tau, p_value = kendalltau(y, y_pred)
    return(tau)

def tau_scoring_p(y, y_pred):
    tau, p_value = kendalltau(y, y_pred)
    return(p_value)

def check_tau(scoring):

    if 'tau' in scoring:
        scoring['tau'] = make_scorer(tau_scoring)
        scoring['tau_p'] = make_scorer(tau_scoring_p)
        print(scoring)

    else:
        scoring = scoring

    return scoring


def do_log(self, x):
    y = np.log(x+1)
    return(y)

def do_exp(self, x):
    y = np.exp(x)-1
    return(y)

def merge_obs_env(obs_path = "../data/gridded_abundances.csv",
                  env_path = "../data/env_data.nc",
                  env_vars = ["temperature", "si", 
                              "phosphate", "din", 
                              "o2", "mld", "DIC", 
                              "TA", "irradiance", 
                              "chlor_a","Rrs_547",
                              "Rrs_667","CI_2",
                              "pic","si_star",
                              "si_n","FID", 
                              "time", "depth", 
                              "lat", "lon"],
                    out_path = "../data/obs_env.csv"):

    d = pd.read_csv(obs_path)

    # #regrid
    # depth_bins = np.linspace(0, 205, 62)
    # depth_labels = np.linspace(0, 300, 61)
    # d['Depth'] = pd.cut(d['Depth'], bins=depth_bins, labels=depth_labels).astype(np.float64) 

    # lat_bins = np.linspace(-90, 90, 181)
    # lat_labels = np.linspace(-90, 89, 180)
    # d['Latitude'] = pd.cut(d['Latitude'].astype(np.float64), bins=lat_bins, labels=lat_labels).astype(np.float64) 

    # lon_bins = np.linspace(-180, 180, 361)
    # lon_labels = np.linspace(-180, 179, 360)
    # d['Longitude'] = pd.cut(d['Longitude'].astype(np.float64), bins=lon_bins, labels=lon_labels).astype(np.float64) 

    #d['DateTime'] = pd.to_datetime(d['Date'],dayfirst=True)
    #d['Month'] = pd.DatetimeIndex(d['DateTime']).month
    #d['Year'] = pd.DatetimeIndex(d['DateTime']).year

    d = d.convert_dtypes()

    d = d.groupby(['Latitude', 'Longitude', 'Depth', 'Month']).mean().reset_index()
    d.rename({'Latitude':'lat','Longitude':'lon','Depth':'depth','Month':'time'},inplace=True,axis=1)
    d.set_index(['lat', 'lon', 'depth', 'time'], inplace=True)

    print("loading env")

    ds = xr.open_dataset(env_path)
    print("converting to dataframe")
    df = ds.to_dataframe()
    ds = None 
    df.reset_index(inplace=True)
    df = df[env_vars]
    df.set_index(['lat','lon','depth','time'],inplace=True)
    print("merging environment")

    out = d.merge(df, how="left", left_index=True, right_index=True)
    out.to_csv(out_path, index=True)
    print("fin")

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
        # #X, y = check_X_y(X, y)
        # #self._check_n_features(X, reset=True)

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

            self.classifier_.fit(X, y != 0)

        non_zero_indices = np.where(self.classifier_.predict(X) == 1)[0]

        if non_zero_indices.size > 0:
            try:
                check_is_fitted(self.regressor)
                self.regressor_ = self.regressor
            except NotFittedError:
                self.regressor_ = clone(self.regressor)

                if isinstance(X, pd.DataFrame):
                    self.regressor_.fit(
                            X.iloc[non_zero_indices],
                            y[non_zero_indices],
                    )
                else:

                    self.regressor_.fit(
                            X[non_zero_indices],
                            y[non_zero_indices],
                    )
        else:
            None

        #self.classifier_ = self.classifier
        #self.regressor_ = self.regressor
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
        #check_is_fitted(self)
#        X = check_array(X)
#        self._check_n_features(X, reset=False)

        output = np.zeros(len(X))

        non_zero_indices = np.where(self.classifier_.predict(X) == 1)[0]

        if non_zero_indices.size > 0:
            if isinstance(X, pd.DataFrame):
                output[non_zero_indices] = self.regressor_.predict(X.iloc[non_zero_indices])
            else:
                output[non_zero_indices] = self.regressor_.predict(X[non_zero_indices])

        return output


class LogGridSearch:
    def __init__(self, m, verbose, cv, param_grid, scoring, regions=None):
        self.m = m
        self.verbose = verbose
        self.cv = cv
        self.param_grid = param_grid
        self.scoring = scoring
        self.regions = regions

    def do_log(self, x):
        y = np.log(x+1)
        return(y)
    
    def do_exp(self, x):
        y = np.exp(x)-1
        return(y)
    
    def transformed_fit(self, X, y, log, predictors):

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
                print("best = nolog")
        
            elif (grid_search1.best_score_ < grid_search2.best_score_):
                grid_search = grid_search2
                best_transformation = "log"
                print("best = log")
            else:
                print("same performance for both models")
                grid_search = grid_search1
                best_transformation = "nobest"
            grid_search.cv_results_['best_transformation'] = best_transformation

        else: 
            print("defined log invalid, pick from yes, no or both")

        return grid_search


class UpsampledZeroStratifiedKFold:
    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        #convert target variable to binary for stratified sampling
        y_binary = np.where(y!=0, 1, 0)

        for rx, tx in StratifiedKFold(n_splits=self.n_splits).split(X,y_binary):
            nix = np.where(y_binary[rx]==0)[0]
            pix = np.where(y_binary[rx]==1)[0]
            pixu = np.random.choice(pix, size=nix.shape[0], replace=True)
            ix = np.append(nix, pixu)
            rxm = rx[ix]
            yield rxm, tx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
    
    
    
class ZeroStratifiedKFold:
    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        #convert target variable to binary for stratified sampling
        y_binary = np.where(y!=0, 1, 0)

        for rx, tx in StratifiedKFold(n_splits=self.n_splits).split(X,y_binary):
            yield rx, tx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
    


def example_data(y_name, n_samples=500, n_features=5, noise=20, random_state=59):

    #example data:
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state)
    # scale so values are strictly positive:
    scaler = MinMaxScaler()  
    scaler.fit(y.reshape(-1,1))  
    y = scaler.transform(y.reshape(-1,1))
    # add exp transformation to data
    # make distribution exponential:
    y = np.exp(y)-1
    #cut tail
    y[y <= 0.5] = 0
    y = np.squeeze(y)
    y = pd.Series(y)
#    y = pd.DataFrame({y_name: y})
    y.name = y_name
    return(X, y)



def lat_weights(d):
    d_w = d*10
    return(d_w)

def calculate_weights(n, mae_list):
    m_values = [-1 * mae for mae in mae_list]
    m = mae_list[n]
    
    mae_sum_m = sum(m_values)/m

    mae_sum = sum(m_values)

    mae_sums_inverse = sum([(mae_sum / mae) for mae in mae_list])
    w = mae_sum_m / mae_sums_inverse
    return(w)


def score_model(m, X_train, y, cv, verbose, scoring, n_jobs):
    scores = cross_validate(m, X_train, y, cv=cv, n_jobs=n_jobs,
                            verbose =verbose, scoring=scoring)
    return(scores)


def def_prediction(path_out, ensemble_config, n, species):

    path_to_scores  = path_out + ensemble_config["m"+str(n+1)] + "/scoring/"
    path_to_param  = path_out +  ensemble_config["m"+str(n+1)] + "/model/"


    if (ensemble_config["classifier"] ==True) and (ensemble_config["regressor"] == False):
        print("predicting classifier")
        m = pickle.load(open(path_to_param + species + '_clf.sav', 'rb'))
        scoring =  pickle.load(open(path_to_scores + species + '_clf.sav', 'rb'))    
        scores = 1-np.mean(scoring['test_accuracy']) #subtract 1 since lower is better

    elif (ensemble_config["classifier"] ==False) and (ensemble_config["regressor"] == True):
        print("predicting regressor")
        m = pickle.load(open(path_to_param + species + '_reg.sav', 'rb'))
        scoring =  pickle.load(open(path_to_scores + species + '_reg.sav', 'rb'))   
        scores = np.mean(scoring['test_MAE'])


    elif (ensemble_config["classifier"] ==True) and (ensemble_config["regressor"] == True):
        print("predicting zero-inflated regressor")
        m = pickle.load(open(path_to_param + species + '_zir.sav', 'rb'))
        scoring =  pickle.load(open(path_to_scores + species + '_zir.sav', 'rb'))    
        scores = np.mean(scoring['test_MAE'])

    elif (ensemble_config["classifier"] ==False) and (ensemble_config["regressor"] == False):

        print("Both regressor and classifier are defined as false")

    return(m, scores)




def export_prediction(m,species, X_predict, model_config, ensemble_config, ens_model_out):

    d = X_predict.copy()
    if (model_config['predict_probability'] == True) and (ensemble_config["regressor"] ==False):
        print("predicting probabilities")
        d[species] = m.predict_proba(X_predict)[:, 1]
    elif (model_config['predict_probability'] == True) and (ensemble_config["regressor"] ==True):
        print("error: can't predict probabilities if the model is a regressor")
    else:
        d[species] = m.predict(X_predict)
    d = d.to_xarray()
    
    try: #make new dir if needed
        os.makedirs(ens_model_out)
    except:
        None

    d[species].to_netcdf(ens_model_out + species + ".nc") 





