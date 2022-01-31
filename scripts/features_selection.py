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

import utils.helpers as helpers
import utils.utils as utils
from data_management.data import Data


def select_features(target_label, no_hist_days, no_hist_weeks, begin_test_date,
                                    scale_data, threshold, save_name_pkl, plot):

    ### Import data
    begin_test_date = "2025-11-06"
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
            if plot:
                print('%.5f ->  %s' % (feat_i, feat_n))
            feats_names_selected.append(feat_n)

    ### Plot feature importance
    if plot:
        plt.figure(figsize=[12,9])
        sns.barplot(x=feats_importances, y=feats_names)
        plt.show()

    ### Save selected features
    with open(os.path.join("output/features", save_name_pkl), "wb") as file:
        pickle.dump(feats_names_selected, file)


def evaluate_model(target_label, no_hist_days, no_hist_weeks, begin_test_date,
                                    scale_data, threshold, save_name_pkl, plot):

    ### Load selected features
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

    regr = AdaptiveRandomForestRegressor(random_state=1, drift_detection_method=None,
                                                        warning_detection_method=None)
    if X_train.shape[1] <= 1 :
        return 1e10
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

    """Plot"""
    if plot:
        plt.figure()
        plt.plot(y_test, label="test")
        plt.plot(y_pred, label="pred")
        plt.legend()
        plt.show()

    return rmse



if __name__ == "__main__":

    params = {
        "target_label" : "new_cases",
        "no_hist_days" : 6,
        "no_hist_weeks" : 0,
        "begin_test_date" : "2021-11-14",
        "scale_data" : None,
        "threshold" : 0.1,
        "save_name_pkl" : "d7w7t0001.pkl",
        "plot": False
    }

    # select_features(**params)
    # params["begin_test_date"] = "2021-11-14"
    # # params["begin_test_date"] = "2020-03-07"  # uncomment for whole stream evaluation
    # print(evaluate_model(**params))

    from itertools import product
    from types import SimpleNamespace
    def dict_configs(param_dict):
        for vcomb in product(*param_dict.values()):
            yield dict(zip(param_dict.keys(), vcomb))

    param_dict = {
        "no_hist_days": [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
        "no_hist_weeks": [0,1,2,3],
        "threshold": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]
    }

    config_best = SimpleNamespace(rmse_best=1e10, params_best=None, rmse_list=[], params_list=[])
    idx_all = len(list(product(*list(param_dict.values()))))

    for idx, params_temp in enumerate(dict_configs(param_dict)):
        print(f"ITER: {idx} / {idx_all}")
        params["no_hist_days"] = params_temp["no_hist_days"]
        params["no_hist_weeks"] = params_temp["no_hist_weeks"]
        params["threshold"] = params_temp["threshold"]

        select_features(**params)
        rmse = evaluate_model(**params)

        config_best.rmse_list.append(rmse)
        config_best.params_list.append(params_temp)

        if rmse < config_best.rmse_best:
            config_best.rmse_best = rmse
            config_best.params_best = params_temp
            print("RMSE:", rmse, " params:", params_temp, "\n")

    sort_idx = np.argsort(config_best.rmse_list)
    config_best.rmse_list = np.array(config_best.rmse_list)[sort_idx]
    config_best.params_list = np.array(config_best.params_list)[sort_idx]

    for idx, (rmse, params) in enumerate(zip(config_best.rmse_list, config_best.params_list)):
        print(rmse, params)
        if idx > 40:
            break

    print("\nBest rmse:", config_best.rmse_best)
    print("Best params:", config_best.params_best)



