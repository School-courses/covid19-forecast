from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from skmultiflow.data import DataStream
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.meta import AdaptiveRandomForestRegressor
from skmultiflow.trees import (HoeffdingTreeRegressor,
                               StackedSingleTargetHoeffdingTreeRegressor,
                               iSOUPTreeRegressor)
from xgboost import XGBRegressor

import utils.helpers as helpers
import utils.utils as utils
from data_management.data import Data

"""Choose model"""
regr = AdaptiveRandomForestRegressor()
# regr = helpers.DummyRegressor()
# regr = helpers.MultiflowPredictorWrapper(SVR())


"""Set optimized parameters"""
# you need to appropriatly set datastream parameters under """define stream parameters"""
model_saved_config = "output/AdaptiveRandomForest/scale"
f = open(model_saved_config + "/report_train.txt", "r")
out = f.read().split("\n")[4]
config = dict(eval(out, {'OrderedDict': OrderedDict}))
no_hist_days = config["data_window_size_days"]
no_hist_weeks = config["data_window_size_weeks"]
scale_data = config["scale_data"]
config.pop("data_window_size_days")
config.pop("data_window_size_weeks")
config.pop("scale_data")
config["random_state"] = None
regr.set_params(**config)


"""define stream parameters"""
target_label = "new_cases"
begin_test_date = "2021-11-06"
# begin_test_date = "2020-03-07"
# no_hist_days = 0
# no_hist_weeks = 1
# scale_data = None
# config = {}


"""import data and initialize stream"""
data = Data(
    no_hist_days=no_hist_days,
    no_hist_weeks=no_hist_weeks,
    target_label=target_label,
    begin_test_date=begin_test_date,
    scale_data=scale_data
)
X_train, y_train, X_test_t, y_test_t = data.get_data()
stream = DataStream(X_test_t, y_test_t)

repetitions = 10
y_pred_list = []
for _ in range(repetitions):
    regr.reset()
    regr.set_params(**config)
    stream.restart()

    """Warm start"""
    regr.fit(X_train, y_train)

    """Partial fit and predict"""
    y_pred, y_test = [], []
    while stream.has_more_samples():
        x_t, y_t = stream.next_sample()
        y_p = regr.predict(x_t)[0]
        regr.partial_fit(x_t, y_t)
        y_pred.append(y_p)
        y_test.append(y_t)

    y_pred = np.array(y_pred).flatten()
    y_test = np.array(y_test).flatten()
    y_pred_list.append(y_pred)

y_pred_avg = np.mean(np.array(y_pred_list), axis=0)
y_pred_std = np.std(np.array(y_pred_list), axis=0)


"""Calculate errors"""
print(model_saved_config)
rmse_list = [mean_squared_error(y_test, y_pred, squared=False) for y_pred in y_pred_list]
rmse_avg = np.mean(rmse_list)
rmse_std = np.std(rmse_list)

print(f"RMSE mean: {rmse_avg}")
print(f"RMSE std: {rmse_std}")

# rmse_base = helpers.calculate_base_rmse(target_label)
# print(f"RRMSE: {rmse/rmse_base}")


"""Plot"""
utils.plot_mean_and_deviations(y_pred_avg, y_pred_std, y_test)


