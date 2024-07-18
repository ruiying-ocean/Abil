import unittest
unittest.TestLoader.sortTestMethodsUsing = None


import sys, os
from yaml import load
from yaml import CLoader as Loader

import pandas as pd
from abil.tune import tune
from abil.functions import example_data, upsample
from abil.predict import predict


# class TestClassifiers(unittest.TestCase):

#     def test_tune_randomforest(self):
#         yaml_path = os.path.abspath(os.path.join(sys.path[0] , os.pardir))

#         with open(yaml_path +'/configuration/classifier_test.yml', 'r') as f:
#             model_config = load(f, Loader=Loader)

#         X_train, y = example_data("Test")

#         m = tune(X_train, y, model_config)
    
#         m.train(model="rf", classifier=True)


#     def test_tune_xgb(self):
#         yaml_path = os.path.abspath(os.path.join(sys.path[0] , os.pardir))

#         with open(yaml_path +'/configuration/classifier_test.yml', 'r') as f:
#             model_config = load(f, Loader=Loader)

#         X_train, y = example_data("Test")

#         m = tune(X_train, y, model_config)
    
#         m.train(model="xgb", classifier=True)


#     def test_tune_knn(self):
#         yaml_path = os.path.abspath(os.path.join(sys.path[0] , os.pardir))

#         with open(yaml_path +'/configuration/classifier_test.yml', 'r') as f:
#             model_config = load(f, Loader=Loader)

#         X_train, y = example_data("Test")

#         m = tune(X_train, y, model_config)
    
#         m.train(model="knn", classifier=True)


#     def test_predict_ensemble(self):
#         yaml_path = os.path.abspath(os.path.join(sys.path[0] , os.pardir))
        
#         with open(yaml_path +'/configuration/classifier_test.yml', 'r') as f:
#             model_config = load(f, Loader=Loader)

#         X_train, y = example_data("Test")
#         X_predict = X_train
        
#         m = predict(X_train, y, X_predict, model_config)
#         m.make_prediction()


#     # def clear_tmp(self):
#     #     yaml_path = os.path.abspath(os.path.join(sys.path[0] , os.pardir))
        
#     #     with open(yaml_path +'/configuration/classifier_test.yml', 'r') as f:
#     #         model_config = load(f, Loader=Loader)

#     #     os.rmdir(model_config['local_root'])
#     #     print("deleted:" + model_config['local_root'])





class TestRegressors(unittest.TestCase):

    def test_tune_randomforest(self):
        yaml_path = os.path.abspath(os.path.join(sys.path[0] , os.pardir))

        with open(yaml_path +'/tests/regressor.yml', 'r') as f:
            model_config = load(f, Loader=Loader)

        model_config['local_root'] = yaml_path
        predictors = model_config['predictors']
        d = pd.read_csv(model_config['local_root'] + model_config['training'])
        target =  "Emiliania huxleyi"
        d = d.dropna(subset=[target])
        d = d.dropna(subset=predictors)
        X_train = d[predictors]
        y = d[target]

        m = tune(X_train, y, model_config)
    
        m.train(model="rf", regressor=True)


    def test_tune_xgb(self):
        yaml_path = os.path.abspath(os.path.join(sys.path[0] , os.pardir))

        with open(yaml_path +'/tests/regressor.yml', 'r') as f:
            model_config = load(f, Loader=Loader)

        model_config['local_root'] = yaml_path
        predictors = model_config['predictors']
        d = pd.read_csv(model_config['local_root'] + model_config['training'])
        target =  "Emiliania huxleyi"
        d = d.dropna(subset=[target])
        d = d.dropna(subset=predictors)
        X_train = d[predictors]
        y = d[target]

        m = tune(X_train, y, model_config)
    
        m.train(model="xgb", regressor=True)


    def test_tune_knn(self):
        yaml_path = os.path.abspath(os.path.join(sys.path[0] , os.pardir))

        with open(yaml_path +'/tests/regressor.yml', 'r') as f:
            model_config = load(f, Loader=Loader)

        model_config['local_root'] = yaml_path
        predictors = model_config['predictors']
        d = pd.read_csv(model_config['local_root'] + model_config['training'])
        target =  "Emiliania huxleyi"
        d = d.dropna(subset=[target])
        d = d.dropna(subset=predictors)
        X_train = d[predictors]
        y = d[target]

        m = tune(X_train, y, model_config)
    
        m.train(model="knn", regressor=True)


    def test_predict_ensemble(self):
        yaml_path = os.path.abspath(os.path.join(sys.path[0] , os.pardir))
        
        with open(yaml_path +'/tests/regressor.yml', 'r') as f:
            model_config = load(f, Loader=Loader)

        model_config['local_root'] = yaml_path
        predictors = model_config['predictors']
        d = pd.read_csv(model_config['local_root'] + model_config['training'])
        target =  "Emiliania huxleyi"


        d = d.dropna(subset=[target])
        d = d.dropna(subset=predictors)

        X_train = d[predictors]
        y = d[target]

        X_predict = X_train
        
        m = predict(X_train, y, X_predict, model_config)
        m.make_prediction()


    # def clear_tmp(self):
    #     yaml_path = os.path.abspath(os.path.join(sys.path[0] , os.pardir))
        
    #     with open(yaml_path +'/configuration/classifier_test.yml', 'r') as f:
    #         model_config = load(f, Loader=Loader)

    #     os.rmdir(model_config['local_root'])
    #     print("deleted:" + model_config['local_root'])



if __name__ == '__main__':
    unittest.main()