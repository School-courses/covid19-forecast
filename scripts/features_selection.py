import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from skmultiflow.data import DataStream
from skmultiflow.meta import AdaptiveRandomForestRegressor
from xgboost import XGBRegressor

import utils.utils as utils
from data_management.data import Data
import utils.helpers as helpers


def select_features(target_label, no_hist_days, no_hist_weeks, begin_test_date, scale_data, threshold, save_name_pkl):

    ### Import data
    data = Data(
        no_hist_days=no_hist_days,
        no_hist_weeks=no_hist_weeks,
        target_label=target_label,
        begin_test_date=begin_test_date,
        scale_data=scale_data
    )
    X, y,_,_ = data.get_data()
    features_names = data.predictors_col_names

    ### Model
    model = XGBRegressor()
    model.fit(X, y)
    importances = model.feature_importances_

    imp_dict = {key: val for key, val in zip(features_names, importances)}
    sorted_feats = sorted(imp_dict.items(), key=lambda x:x[1], reverse=True)
    feats_names = [x[0] for x in sorted_feats]
    feats_importances = [x[1] for x in sorted_feats]

    ### Importances
    # print feature importance
    feats_names_selected = []
    for feat_n, feat_i in zip(feats_names, feats_importances):
        if feat_i > threshold:
            print('%.5f ->  %s' % (feat_i, feat_n))
            feats_names_selected.append(feat_n)

    ### Plot feature importance
    # plt.figure(figsize=[12,9])
    # sns.barplot(x=feats_importances, y=feats_names)
    # plt.show()

    ### Save selected features
    with open(os.path.join("output/features", save_name_pkl), "wb") as file:
        pickle.dump(feats_names_selected, file)


def evaluate_model(target_label, no_hist_days, no_hist_weeks, begin_test_date, scale_data, threshold, save_name_pkl):

    with open(os.path.join("output/features", save_name_pkl), "rb") as file:
        selected_features = pickle.load(file)

    ### Import data
    data = Data(
        no_hist_days=no_hist_days,
        no_hist_weeks=no_hist_weeks,
        target_label=target_label,
        begin_test_date=begin_test_date,
        scale_data=scale_data
    )
    data.predictors_col_names = selected_features

    X_train, y_train, X_test_t, y_test_t = data.get_data()
    stream = DataStream(X_test_t, y_test_t)

    regr = AdaptiveRandomForestRegressor()
    regr.fit(X_train, y_train)

    y_pred, y_test = [], []
    while stream.has_more_samples():
        x_t, y_t = stream.next_sample()
        y_p = regr.predict(x_t)[0]
        regr.partial_fit(x_t, y_t)
        y_pred.append(y_p)
        y_test.append(y_t)

    y_pred = np.array(y_pred).flatten()
    y_test = np.array(y_test).flatten()


    rmse = mean_squared_error(y_test, y_pred, squared=False)
    # print(f"RMSE: {rmse}")
    return rmse

    """Plot"""
    # plt.figure()
    # plt.plot(y_test, label="test")
    # plt.plot(y_pred, label="pred")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":

    params = {
        "target_label" : "new_cases",
        "no_hist_days" : 15,
        "no_hist_weeks" : 5,
        "begin_test_date" : "2025-11-06",
        "scale_data" : None,
        "threshold" : 0.1,
        "save_name_pkl" : "d7w7t0001.pkl"
    }

    # select_features(**params)
    # params["begin_test_date"] = "2021-11-14"
    # # params["begin_test_date"] = "2020-03-07"  # uncomment for whole stream evaluation
    # evaluate_model(**params)

    from itertools import product

    def dict_configs(d):
        for vcomb in product(*d.values()):
            yield dict(zip(d.keys(), vcomb))

