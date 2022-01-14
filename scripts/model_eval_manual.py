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
regr = AdaptiveRandomForestRegressor(random_state=1)
# regr = helpers.MultiflowPredictorWrapper(SVR())


"""Set optimized parameters"""
## you need to appropriatly set datastream parameters under """define stream parameters"""
model_saved_config = "output/AdaptiveRandomForest/500adw"
f = open(model_saved_config + "/report_train.txt", "r")
out = f.read().split("\n")[4]
config = dict(eval(out, {'OrderedDict': OrderedDict}))
no_hist_vals = config["data_window_size"]
config.pop("data_window_size")
config["drift_detection_method"] = ADWIN(config["drift_detection_method"])
config["warning_detection_method"] = ADWIN(config["warning_detection_method"])
regr.set_params(**config)


"""define stream parameters"""
target_label = "new_cases"
# no_hist_vals = 9
begin_test_date = "2021-11-06"


"""import data and initialize stream"""
data = Data(no_hist_vals, target_label, begin_test_date=begin_test_date)
X_train, y_train, X_test_t, y_test_t = data.get_data()
stream = DataStream(X_test_t, y_test_t)


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

print("y_pred", y_pred.shape)
print("y_test", y_test.shape)


"""Calculate errors"""
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse}")
rmse_base = helpers.calculate_base_rmse(target_label)
print(f"RRMSE: {rmse/rmse_base}")


"""Plot"""
plt.figure()
plt.plot(y_test)
plt.plot(y_pred)
plt.show()
