import unittest
unittest.TestLoader.sortTestMethodsUsing = None

import sys, os
from yaml import load
from yaml import CLoader as Loader
import pandas as pd
import numpy as np
import xarray as xr 

from abil.tune import tune
from abil.utils import example_data # example_training_data, example_predict_data
from abil.predict import predict
from abil.post import post

class TestRegressors(unittest.TestCase):

    def setUp(self):
        self.workspace = os.getenv('GITHUB_WORKSPACE', '.')
        with open(os.path.join(self.workspace,'tests/regressor.yml'), 'r') as f:
            self.model_config = load(f, Loader=Loader)

        self.model_config['local_root'] = self.workspace # yaml_path


        self.target_name =  "Emiliania huxleyi"
        self.X_train, self.X_predict, self.y = example_data(self.target_name, n_samples=1000, n_features=3, noise=0.1, train_to_predict_ratio=0.7, random_state=59)
#        self.X_predict = X_predict[predictors]


    def test_post_ensemble(self):
        m = tune(self.X_train, self.y, self.model_config)
        m.train(model="rf", log="yes")
        m.train(model="xgb", log="yes")
        m.train(model="knn", log="yes")

        m = predict(self.X_train, self.y, self.X_predict, self.model_config, n_jobs=self.model_config['n_threads'])
        m.make_prediction()

        # Load datasets
        rf = xr.open_dataset("./tests/ModelOutput/regressor/predictions/rf/Emiliania_huxleyi.nc")
        xgb = xr.open_dataset("./tests/ModelOutput/regressor/predictions/xgb/Emiliania_huxleyi.nc")  # Note: same path as rf?

        # Calculate sums
        rf_mean_sum = np.sum(rf['mean'])
        rf_ci95_UL_sum = np.sum(rf['ci95_UL'])
        xgb_mean_sum = np.sum(xgb['mean'])
        xgb_ci95_UL_sum = np.sum(xgb['ci95_UL'])

        print("======================")
        print("DEBUG OF PREDICT SUMS")
        print("======================")

        print(f"RF mean sum: {rf_mean_sum}")
        print(f"XGB mean sum: {xgb_mean_sum}")
        # Check if values are within same order of magnitude
#        if not np.isclose(rf_mean_sum, xgb_mean_sum, rtol=9):
#            raise AssertionError("Mean sums are not within an order of magnitude")
        
        print(f"RF ci95_UL sum: {rf_ci95_UL_sum}")
        print(f"XGB ci95_UL sum: {xgb_ci95_UL_sum}")
#        if not np.isclose(rf_ci95_UL_sum, xgb_ci95_UL_sum, rtol=9):
#            raise AssertionError("CI95 UL sums are not within an order of magnitude")

        targets = np.array([self.target_name])
        def do_post(statistic):
            m = post(self.X_train, self.y, self.X_predict, self.model_config, statistic, datatype="poc")
            #estimate aoa for each target and export to aoa.nc:
            m.estimate_applicability()

            m.estimate_carbon("pg poc")

            m.total()

            m.merge_env()
            m.merge_obs("test",targets)

            m.export_ds("test")
            m.export_csv("test")

            vol_conversion = 1e3 #L-1 to m-3
            integ = m.integration(m, vol_conversion=vol_conversion)
            print(targets)
            integ.integrated_totals(targets)
            integ.integrated_totals(targets, monthly=True)

        do_post(statistic="mean")
        do_post(statistic="ci95_UL")
        do_post(statistic="ci95_LL")




class Test2Phase(unittest.TestCase):

    def setUp(self):
        self.workspace = os.getenv('GITHUB_WORKSPACE', '.')
        with open(os.path.join(self.workspace, 'tests/2-phase.yml'), 'r') as f:
            self.model_config = load(f, Loader=Loader)

        self.model_config['local_root'] = self.workspace # yaml_path


        self.target_name =  "Emiliania huxleyi"

        self.X_train, self.X_predict, self.y = example_data(self.target_name, n_samples=1000, n_features=3, noise=0.1, train_to_predict_ratio=0.7, random_state=59)
        

    def test_post_ensemble(self):


        m = tune(self.X_train, self.y, self.model_config)

        m.train(model="rf", log="yes")
        m.train(model="xgb", log="yes")
        m.train(model="knn", log="yes")

        m = predict(self.X_train, self.y, self.X_predict, self.model_config, n_jobs=self.model_config['n_threads'])
        m.make_prediction()
        print("======================")
        print("DEBUG OF PREDICT SUMS")
        print("======================")
        # Load datasets
        rf = xr.open_dataset("./tests/ModelOutput/2-phase/predictions/rf/Emiliania_huxleyi.nc")
        xgb = xr.open_dataset("./tests/ModelOutput/2-phase/predictions/xgb/Emiliania_huxleyi.nc")  # Note: same path as rf?

        # Calculate sums
        rf_mean_sum = np.sum(rf['mean'])
        rf_ci95_UL_sum = np.sum(rf['ci95_UL'])
        xgb_mean_sum = np.sum(xgb['mean'])
        xgb_ci95_UL_sum = np.sum(xgb['ci95_UL'])

        print(f"RF mean sum: {rf_mean_sum}")
        print(f"XGB mean sum: {xgb_mean_sum}")
        
        # Check if values are within same order of magnitude
#        if not np.isclose(rf_mean_sum, xgb_mean_sum, rtol=9):
#            raise AssertionError("Mean sums are not within an order of magnitude")
        
        print(f"RF ci95_UL sum: {rf_ci95_UL_sum}")
        print(f"XGB ci95_UL sum: {xgb_ci95_UL_sum}")
#        if not np.isclose(rf_ci95_UL_sum, xgb_ci95_UL_sum, rtol=9):
#            raise AssertionError("CI95 UL sums are not within an order of magnitude")

        targets = np.array([self.target_name])

        def do_post(statistic):
            m = post(self.X_train, self.y, self.X_predict, self.model_config, statistic, datatype="poc")
            #estimate aoa for each target and export to aoa.nc:
            m.estimate_applicability()
            m.estimate_carbon("pg poc")
            m.diversity()

            m.total()
            m.merge_env()
            m.merge_obs("test",targets)

            m.export_ds("test")

            vol_conversion = 1e3 #L-1 to m-3
            integ = m.integration(m, vol_conversion=vol_conversion)
            integ.integrated_totals(targets, monthly=True)
            integ.integrated_totals(targets)

        do_post(statistic="mean")
        do_post(statistic="ci95_UL")
        do_post(statistic="ci95_LL")

if __name__ == '__main__':
    # Create a test suite combining all test cases in order
    suite = unittest.TestSuite()
    suite.addTest(TestRegressors('test_post_ensemble'))
    suite.addTest(Test2Phase('test_post_ensemble'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
