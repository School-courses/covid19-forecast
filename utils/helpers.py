
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_relation_table(stream, stream_day, no_hist_vals):
    """ Transform stream to relational/attricutional form

    Args:
        stream (list): list of consequtive number of stream
        stream_day (list) day of the week
        no_hist_vals (int): number of historical values to include in predictions
    """
    X = []
    y = []
    y_day = []
    X_temp = []
    for val, day in zip(stream, stream_day):
        X_temp.append(val)
        if len(X_temp) == no_hist_vals:
            if len(X) != 0:
                y.append(val)
                y_day.append(day)

            X.append(X_temp.copy())
            X_temp.pop(0)
    X.pop(-1)

    return X, y, y_day


def calculate_base_rmse(y_label):
    data_pd = pd.read_csv("data/slovenia_weekly.csv")
    stream = data_pd[y_label].to_numpy()
    X, y, _ = create_relation_table(stream, stream, 4)

    y_pred = []
    for x in X:
        y_pred.append(np.mean(x))

    rmse_base = mean_squared_error(y, y_pred, squared=False)
    return rmse_base


class DummyRegressor():
    def predict(self, x):
        return [np.mean(x)]
    def partial_fit(self, x, y):
        pass
    def set_params(self, **kwargs):
        pass
    def fit(self, X, y):
        pass
    def reset(self):
        pass


class MultiflowPredictorWrapper():
    def __init__(self, model_class):
        self.model_class = model_class
        self.model = None
        self.X = None
        self.y = None

    def fit(self, X_train, y_train):
        self.partial_fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def partial_fit(self, X_train, y_train):
        if self.X is None:
            self.X = X_train
            self.y = y_train
        else:
            self.X = np.vstack([self.X, X_train])
            self.y = np.vstack([self.y, y_train])
        self.model.fit(self.X, self.y.ravel())

    def set_params(self, **params):
        self.model.set_params(**params)

    def reset(self):
        self.X = None
        self.y = None
        self.model = clone(self.model_class())


def best_window_sum(x, y, max_window):
    # code: https://github.com/julianikulski/bike-sharing/blob/master/bike_sharing_demand.ipynb
    # get list for all rolling sums between categorical feature with numberical target
    corr_temp_cust = []
    for i in range(1, max_window):
        roll_val = list(x.rolling(i).sum()[i-1:-1])
        total_cust_ti = list(y[i:])
        corr, p_val = pearsonr(total_cust_ti, roll_val)
        corr_temp_cust.append(corr)
    # get the optimal window size for rolling mean
    max_val = np.argmax(corr_temp_cust)
    min_val = np.argmin(corr_temp_cust)
    opt_corr_min = corr_temp_cust[min_val]
    opt_corr_max = corr_temp_cust[max_val]
    results = {max_val+1: opt_corr_max, min_val+1: opt_corr_min}
    return results


def best_window_mean(x, y, max_window):
    # code: https://github.com/julianikulski/bike-sharing/blob/master/bike_sharing_demand.ipynb
    # get list for all correlations between a feature and target with different rolling means
    corr_temp_cust = []
    for i in range(1, max_window):
        roll_val = list(x.rolling(i).mean()[i-1:-1])
        total_cust_ti = list(y[i:])
        corr, p_val = pearsonr(total_cust_ti, roll_val)
        corr_temp_cust.append(corr)
    # get the optimal window size for rolling mean between a feature and target
    max_val = np.argmax(corr_temp_cust)
    min_val = np.argmin(corr_temp_cust)
    opt_corr_min = corr_temp_cust[min_val]
    opt_corr_max = corr_temp_cust[max_val]
    results = {max_val+1: opt_corr_max, min_val+1: opt_corr_min}
    return results


def best_window_std(x, y, max_window):
    # code: https://github.com/julianikulski/bike-sharing/blob/master/bike_sharing_demand.ipynb
    # get list for all correlations between a feature and target with different rolling standard deviations
    corr_temp_cust = []
    for i in range(2, max_window):
        roll_val = list(x.rolling(i).std()[i-1:-1])
        total_cust_ti = list(y[i:])
        corr, p_val = pearsonr(total_cust_ti, roll_val)
        corr_temp_cust.append(corr)
    # get the optimal window size for rolling std between a feature and target
    max_val = np.argmax(corr_temp_cust)
    min_val = np.argmin(corr_temp_cust)
    opt_corr_min = corr_temp_cust[min_val]
    opt_corr_max = corr_temp_cust[max_val]
    results = {max_val+1: opt_corr_max, min_val+1: opt_corr_min}
    return results
