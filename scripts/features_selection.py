import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot
from xgboost import XGBRegressor

from data_management.data import Data


def main():
    ### Import data
    target_label = "new_cases"
    no_hist_days = 7
    no_hist_weeks = 7
    begin_test_date = "2025-11-06"
    scale_data = None

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

    # print feature importance
    threshold = 0.0001
    feats_names_selected = []
    for feat_n, feat_i in zip(feats_names, feats_importances):
        if feat_i > threshold:
            print('%.5f ->  %s' % (feat_i, feat_n))
            feats_names_selected.append(feat_n)

    # plot feature importance
    plt.figure(figsize=[12,9])
    sns.barplot(x=feats_importances, y=feats_names)
    plt.show()


    ### Save selected features
    save_name_pkl = "d7w7t0001.pkl"
    with open(os.path.join("output/features", save_name_pkl), "wb") as file:
        pickle.dump(feats_names_selected, file)


if __name__ == "__main__":
    main()
