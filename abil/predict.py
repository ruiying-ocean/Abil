import pandas as pd
import numpy as np
import pickle
import os
import time
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.model_selection import KFold
from mapie.regression import MapieRegressor
from mapie.classification import  MapieClassifier
from mapie.conformity_scores import GammaConformityScore

if 'site-packages' in __file__:
    from abil.functions import calculate_weights, score_model, def_prediction, export_prediction, ZeroStratifiedKFold,  UpsampledZeroStratifiedKFold, check_tau
else:
    from functions import calculate_weights, score_model, def_prediction, export_prediction, ZeroStratifiedKFold,  UpsampledZeroStratifiedKFold, check_tau

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
                
        `cv` : int, number of cross-folds

        `n_threads`: int, number of threads to use
                
        `ensemble_config` : 
        
        `clf_scoring` :
        
        `reg_scoring` :

    """
    def __init__(self, X_train, y, X_predict, model_config, n_jobs=1):
        
        self.st = time.time()

        self.y = y.sample(frac=1, random_state=model_config['seed']) #shuffle
        self.X_train = X_train.sample(frac=1, random_state=model_config['seed']) #shuffle

        self.seed = model_config['seed']
        self.species = y.name
        self.verbose = model_config['verbose']

        if model_config['hpc']==False:
            self.path_out = model_config['local_root'] + model_config['path_out'] 
        elif model_config['hpc']==True:
            self.path_out = model_config['hpc_root'] + model_config['path_out'] 
        else:
            raise ValueError("hpc True or False not defined in yml")
            
        if model_config['stratify']==True:
            if model_config['upsample']==True:
                self.cv = UpsampledZeroStratifiedKFold(n_splits=model_config['cv'])
                print("upsampling = True")
            else:
                self.cv = ZeroStratifiedKFold(n_splits=model_config['cv'])
        else:
            self.cv = KFold(n_splits=model_config['cv'])

        self.X_predict = X_predict
        X_predict = None
        self.ensemble_config = model_config['ensemble_config']
        self.model_config = model_config

        self.n_jobs = n_jobs

        if (self.ensemble_config["classifier"] ==True) and (self.ensemble_config["regressor"] == False):
            self.scoring = model_config['clf_scoring']
            self.y[self.y > 0] = 1

        elif (self.ensemble_config["classifier"] ==False) and (self.ensemble_config["regressor"] == False):
            raise ValueError("classifier and regressor can't both be False")
        else:
            self.scoring = check_tau(self.model_config['reg_scoring']) 


        if (self.ensemble_config["classifier"] !=True) and (self.ensemble_config["classifier"] !=False):
            raise ValueError("classifier should be True or False")
        
        if (self.ensemble_config["regressor"] !=True) and (self.ensemble_config["regressor"] !=False):
            raise ValueError("regressor should be True or False")

        print("initialized prediction")
        
    def make_prediction(self, prediction_inference=True, alpha=[0.32],
                        conformity_score = GammaConformityScore()):

        """
        Calculates performance of model(s) and exports prediction(s) to netcdf

        If more than one model is defined an weighted ensemble is generated using 
        a voting Regressor or Classifier.

        To determine model error Prediction Inference is implemented using MAPIE.

        Parameters
        ----------
        prediction_inference: Optional[bool]
            Whether or not to include prediction inference.
            
        alpha: Optional[float]
            Must be float between ``0`` and ``1``, represents the uncertainty of the
            confidence interval.
            Lower ``alpha`` produce larger (more conservative) prediction
            intervals.
            ``alpha`` is the complement of the target coverage level.

            By default ``[0.32]`.

        conformity_score: Optional[ConformityScore]
            ConformityScore instance.
            It defines the link between the observed values, the predicted ones
            and the conformity scores. 

            - ConformityScore: any MAPIE ``ConformityScore`` class

            By default ``GammaConformityScore()``.


        Notes
        -----
        If more than one model is provided, predictions are made for both 
        invidiual models and an ensemble of the models. 

        For MAPIE: A GammaConformityScore assumes there is no zeros or negative values in Y.
        If this is not the case,  AbsoluteConformityScore() should be used.
        """

        number_of_models = len(self.ensemble_config) -2
        print("number of models in ensemble:" + str(number_of_models))

        if number_of_models==1:

            m, mae1 = def_prediction(self.path_out, self.ensemble_config, 0, self.species)

            model_name = self.ensemble_config["m" + str(1)]
            model_out = self.path_out + model_name + "/predictions/" 
            export_prediction(m, self.species, self.X_predict, self.model_config, self.ensemble_config, model_out)

        elif number_of_models >=2:
                    
            # iteratively make prediction for each model
            models = []
            mae_list = []
            w = []

            for i in range(number_of_models):
                m, mae = def_prediction(self.path_out, self.ensemble_config, i, self.species)
                model_name = self.ensemble_config["m" + str(i + 1)]
                model_out = self.path_out + model_name + "/predictions/" 
                export_prediction(m, self.species, self.X_predict, self.model_config, self.ensemble_config, model_out)

                print("exporting " + model_name + " prediction to: " + model_out)

                models.append((model_name, m))
                mae_list.append(mae)

        
            w = [calculate_weights(i, mae_list) for i in range(len(mae_list))]

            if self.ensemble_config["regressor"] ==True:
                m = VotingRegressor(estimators=models, weights=w).fit(self.X_train, self.y)
                mapie = MapieRegressor(m) #            
            else:
                m= VotingClassifier(estimators=models, weights=w).fit(self.X_train, self.y)
                mapie = MapieClassifier(m) #, conformity_score=conformity_score

            print(np.min(self.y))

            if prediction_inference==True:
                print()
                mapie.fit(self.X_train, self.y)

                low_name = str(int(alpha[0]*100))
                up_name = str(int(np.round((1-alpha[0])*100)))

                print(up_name)
                y_pred = []
                y_pis = []
                i, chunksize = 0, 1000
                for idx in range(0, len(self.X_predict), chunksize):
                    pred, pis = mapie.predict(self.X_predict[idx:(i+1)*chunksize], alpha=alpha)
                    y_pred += list(pred)
                    y_pis += list(pis)
                    i += 1
                    
                y_pred = np.array(y_pred)
                y_pis = np.array(y_pis)

                low_model_out = self.path_out + "mapie/predictions/" + low_name +"/"
                ci50_model_out = self.path_out + "mapie/predictions/50/"
                up_model_out = self.path_out + "mapie/predictions/" + up_name +"/"
                
                try: #make new dir if needed
                    os.makedirs(low_model_out)
                except:
                    None
                try: #make new dir if needed
                    os.makedirs(ci50_model_out)
                except:
                    None
                try: #make new dir if needed
                    os.makedirs(up_model_out)
                except:
                    None

                d_ci50 = self.X_predict.copy()
                d_ci50[self.species] = y_pred
                d_ci50 = d_ci50.to_xarray()
                d_ci50[self.species].to_netcdf(ci50_model_out + self.species + ".nc") 
                print("exported MAPIE CI50 prediction to: " + ci50_model_out + self.species + ".nc")
                d_ci50 = None
                y_pred = None

                d_low = self.X_predict.copy()
                print("min 68 CI:")
                print(np.mean(y_pis[:,0,:].flatten()))
                print("min 68 CI (incl NA):")
                print(np.nanmean(y_pis[:,0,:].flatten()))

                d_low[self.species] = y_pis[:,0,:].flatten()
                d_low[self.species].to_xarray().to_netcdf(low_model_out + self.species + ".nc") 
                print("exported MAPIE " + low_name + " prediction to: " + low_model_out + self.species + ".nc")
                d_low = None

                d_up = self.X_predict.copy()
                d_up[self.species] = y_pis[:,1,:].flatten()
                d_up[self.species].to_xarray().to_netcdf(up_model_out + self.species + ".nc") 
                print("exported MAPIE " + up_name + " prediction to: " + up_model_out + self.species + ".nc")
                d_up = None

            scores = score_model(m, self.X_train, self.y, self.cv, self.verbose, self.scoring, self.n_jobs)

            model_out_scores = self.path_out + "ens/scoring/"
            try: #make new dir if needed
                os.makedirs(model_out_scores)
            except:
                None
            pickle.dump(scores, open(model_out_scores + self.species + '.sav', 'wb'))   
            print("exporting ensemble scores to: " + model_out_scores)

        else:
            raise ValueError("at least one model should be defined in the ensemble")

        et = time.time()
        elapsed_time = et-self.st
        print("finished")
        print("execution time:", elapsed_time, "seconds")