import os

import numpy as np
import pandas as pd


class Data():
    def __init__(self, no_hist_values, target_label, root_dir="", begin_test_date=None):
        data_daily = os.path.join(root_dir, "data/slovenia_daily.csv")
        data_weekly = os.path.join(root_dir, "data/slovenia_weekly.csv")
        self.df_daily = pd.read_csv(data_daily)
        self.df_weekly = pd.read_csv(data_weekly)

        self.df_daily['date'] = pd.to_datetime(self.df_daily['date'])
        self.df_weekly['date'] = pd.to_datetime(self.df_weekly['date'])

        self.target_label = target_label
        self.no_hist_vals = no_hist_values
        self.begin_test_date = begin_test_date
        self.predictors_col_names = []

        self.df_data_table = self.create_base_table()


    def create_base_table(self):
        list_hist_vals = []
        for timestamp in self.df_weekly['date']:
            end_date = timestamp - pd.DateOffset(6) # 6 day offset for week prediction
            start_date = end_date - pd.DateOffset(self.no_hist_vals)
            mask_dates = (self.df_daily["date"] > start_date) & (self.df_daily["date"] <= end_date)
            predictors = self.df_daily[mask_dates].loc[:, self.target_label].to_numpy()
            list_hist_vals.append(predictors)

        bool_hist_vals = [len(hist_vals) == self.no_hist_vals for hist_vals in list_hist_vals]
        num = len(list_hist_vals) - sum(bool_hist_vals)

        # remove num instances which are not equal to self.no_hist_vals
        target_df = self.df_weekly[self.target_label].loc[num:].reset_index(drop=True).to_frame("target")
        date_target_df = self.df_weekly["date"].loc[num:].reset_index(drop=True).to_frame()

        col_names = [f"x_{idx}" for idx in reversed(range(1, len(list_hist_vals[num])+1))]
        predictors_df = pd.DataFrame(list_hist_vals[num:], columns=col_names)
        self.predictors_col_names.extend(col_names)

        df_data_table = pd.concat([date_target_df, target_df, predictors_df], axis=1)
        return df_data_table


    def get_data(self, save=False):
        if save:
            self.df_data_table.to_csv("data/data_table.csv", index=False)

        def convert_to_numpy(df):
            X = df.loc[:, self.predictors_col_names].to_numpy()
            y = df.loc[:, "target"].to_numpy()
            y = np.expand_dims(y, axis = 1)
            return X, y

        df_train = self.df_data_table[self.df_data_table["date"] < self.begin_test_date]
        df_test = self.df_data_table[self.df_data_table["date"] >= self.begin_test_date]

        X_train, y_train = convert_to_numpy(df_train)
        X_test, y_test = convert_to_numpy(df_test)

        return X_train, y_train, X_test, y_test


