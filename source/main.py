import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skmultiflow.meta import AdaptiveRandomForestRegressor

import utils.helpers as helpers
import utils.utils as utils
from data_management.data import Data

target_label = "new_cases"
no_hist_vals = 4

data = Data(no_hist_vals, target_label)
X, y = data.get_data()

num_pretrain = 5
X_train = np.array(X[0:num_pretrain])
y_train = np.array(y[0:num_pretrain])
X_test = np.array(X[num_pretrain:])
y_test = np.array(y[num_pretrain:])

regr = AdaptiveRandomForestRegressor(random_state=1)
regr.fit(X_train, y_train) # warm start

# regr = helpers.DummyRegressor()

y_pred = []
for x_t, y_t in zip(X_test, y_test):
    y_p = regr.predict([x_t])[0]
    y_pred.append(y_p)
    regr.partial_fit([x_t], [y_t])

y_pred = np.array(y_pred)

mse = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"MSE {mse}")
print(f"RMSE: {rmse}")
rmse_base = helpers.calculate_base_rmse(target_label)
print(f"RRMSE: {rmse/rmse_base}")

plt.figure()
utils.plot_data("Prediction", y_test, y_pred)
plt.show()


