import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skmultiflow.data import DataStream
from skmultiflow.lazy import KNNRegressor
from skmultiflow.meta import AdaptiveRandomForestRegressor
from skmultiflow.trees import (HoeffdingTreeRegressor,
                               StackedSingleTargetHoeffdingTreeRegressor,
                               iSOUPTreeRegressor)

import utils.helpers as helpers
import utils.utils as utils
from data_management.data import Data

"""define parameters"""
target_label = "new_cases"
no_hist_vals = 4


"""import data and initialize stream"""
data = Data(no_hist_vals, target_label)
X, y = data.get_data()
stream = DataStream(X, y)


"""Choose model"""
regr = AdaptiveRandomForestRegressor(random_state=1)
regr = KNNRegressor()
regr = HoeffdingTreeRegressor()
regr = StackedSingleTargetHoeffdingTreeRegressor(random_state=1)
regr = iSOUPTreeRegressor()
# regr = helpers.DummyRegressor()


"""Partial fit and predict"""
y_pred, y_test = [], []
while stream.has_more_samples():
    x_t, y_t = stream.next_sample()
    y_p = regr.predict(x_t)[0]
    regr.partial_fit(x_t, y_t)
    y_pred.append(y_p)
    y_test.append(y_t)

y_pred = np.array(y_pred)
y_test = np.array(y_test)

print("y_pred", y_pred.shape)
print("y_test", y_test.shape)

"""Calculate errors"""
mse = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"MSE {mse}")
print(f"RMSE: {rmse}")
rmse_base = helpers.calculate_base_rmse(target_label)
print(f"RRMSE: {rmse/rmse_base}")


"""Plot"""
plt.figure()
utils.plot_data("Prediction", y_test, y_pred)
plt.show()


