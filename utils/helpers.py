
import numpy as np
import pandas as pd
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
        return [np.sum(x)]
    def partial_fit(self, x, y):
        pass


class MultiflowPredictorWrapper():
    def __init__(self, model):
        self.model = model
        # self.model = Pipeline([('scaler', StandardScaler()), ('model', self.model)])
        self.X = None
        self.y = None

    def fit(self, X_train, y_train):
        self.partial_fit(X_train, y_train)
        self.model.fit(self.X, self.y)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def partial_fit(self, X_train, y_train):
        if self.X is None:
            self.X = X_train
            self.y = y_train
        else:
            self.X = np.vstack([self.X, X_train])
            self.y = np.vstack([self.y, y_train])

    def set_params(self, **params):
        self.model.set_params(**params)
