from sklearn.base import BaseEstimator, RegressorMixin, clone, is_regressor, is_classifier
from scipy.stats import kendalltau
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.exceptions import NotFittedError
from inspect import signature
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_regression


def tau_scoring(y, y_pred):
    tau, p_value = kendalltau(y, y_pred)
    return(tau)

def tau_scoring_p(y, y_pred):
    tau, p_value = kendalltau(y, y_pred)
    return(p_value)


def do_log(self, x):
    y = np.log(x+1)
    return(y)

def do_exp(self, x):
    y = np.exp(x)-1
    return(y)


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
            output[non_zero_indices] = self.regressor_.predict(X[non_zero_indices]).ravel()

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