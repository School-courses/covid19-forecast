import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from skmultiflow.data import DataStream
from skmultiflow.meta import AdaptiveRandomForestRegressor
from skmultiflow.trees import (HoeffdingTreeRegressor,
                               StackedSingleTargetHoeffdingTreeRegressor,
                               iSOUPTreeRegressor)
from xgboost import XGBRegressor

import utils.helpers as helpers
import utils.utils as utils
from data_management.data import Data

"""define parameters"""
target_label = "new_cases"
no_hist_vals = 4
begin_test_date = "2021-11-06"


"""import data and initialize stream"""
data = Data(no_hist_vals, target_label, begin_test_date=begin_test_date)
X_train, y_train, X_test, y_test = data.get_data()
stream = DataStream(X_test, y_test)


"""Choose model"""
regr = AdaptiveRandomForestRegressor(random_state=1)
regr = helpers.MultiflowPredictorWrapper(SVR())


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

y_pred = np.array(y_pred)
y_test = np.array(y_test)

print("y_pred", y_pred.shape)
print("y_test", y_test.shape)


"""Calculate errors"""
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse}")
rmse_base = helpers.calculate_base_rmse(target_label)
print(f"RRMSE: {rmse/rmse_base}")


"""Plot"""
plt.figure()
utils.plot_data("Prediction", y_test, y_pred)
plt.show()


