import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale, scale


class Data():
    def __init__(self, no_hist_days, no_hist_weeks, target_label, root_dir="", begin_test_date=None, scale_data=None):
        data_daily = os.path.join(root_dir, "data/slovenia_daily.csv")
        data_weekly = os.path.join(root_dir, "data/slovenia_weekly.csv")
        self.df_daily = pd.read_csv(data_daily)
        self.df_weekly = pd.read_csv(data_weekly)

        self.df_daily['date'] = pd.to_datetime(self.df_daily['date'])
        self.df_weekly['date'] = pd.to_datetime(self.df_weekly['date'])

        self.target_label = target_label
        self.no_hist_days = no_hist_days
        self.no_hist_weeks = no_hist_weeks
        self.begin_test_date = begin_test_date
        self.scale_data = scale_data
        self.predictors_col_names = []

        self.df_data_table = self._create_data_table()


    def _create_base_table(self):
        """ Create base table filled with daily values, target value and date"""
        list_hist_vals = []
        for timestamp in self.df_weekly['date']:
            end_date = timestamp - pd.DateOffset(7) # 7 day (1 week) offset for week prediction
            start_date = end_date - pd.DateOffset(self.no_hist_days)
            mask_dates = (self.df_daily["date"] > start_date) & (self.df_daily["date"] <= end_date)
            predictors = self.df_daily[mask_dates].loc[:, self.target_label].to_numpy()
            list_hist_vals.append(predictors)

        bool_hist_vals = [len(hist_vals) == self.no_hist_days for hist_vals in list_hist_vals]
        num = len(list_hist_vals) - sum(bool_hist_vals)

        # remove num instances which are not equal to self.no_hist_vals
        target_df = self.df_weekly[self.target_label].loc[num:].reset_index(drop=True).to_frame("target")
        date_target_df = self.df_weekly["date"].loc[num:].reset_index(drop=True).to_frame()

        col_names = [f"d_{idx}" for idx in reversed(range(1, len(list_hist_vals[num])+1))]
        predictors_df = pd.DataFrame(list_hist_vals[num:], columns=col_names)
        self.predictors_col_names.extend(col_names)

        df_base_table = pd.concat([date_target_df, target_df, predictors_df], axis=1)
        return df_base_table


    def _create_weekly_table(self, df_base_table):
        """ Create table with weekly values"""
        list_hist_vals = []
        for timestamp in df_base_table['date']:
            end_date = timestamp - pd.DateOffset(weeks=1) # 1 week offset for prediction
            start_date = end_date - pd.DateOffset(weeks=self.no_hist_weeks)
            mask_dates = (self.df_weekly["date"] > start_date) & (self.df_weekly["date"] <= end_date)
            predictors = self.df_weekly[mask_dates].loc[:, self.target_label].to_numpy()
            list_hist_vals.append(predictors)

        bool_hist_vals = [len(hist_vals) == self.no_hist_weeks for hist_vals in list_hist_vals]
        num = len(list_hist_vals) - sum(bool_hist_vals)
        date_target_df = df_base_table["date"].loc[num:].reset_index(drop=True).to_frame()

        col_names = [f"w_{idx}" for idx in reversed(range(1, len(list_hist_vals[num])+1))]
        predictors_df = pd.DataFrame(list_hist_vals[num:], columns=col_names)
        self.predictors_col_names.extend(col_names)

        df_weekly_table = pd.concat([date_target_df, predictors_df], axis=1)
        return df_weekly_table


    def _create_data_table(self):
        base_table = self._create_base_table()
        weekly_table = self._create_weekly_table(base_table)
        data_table = pd.merge(base_table, weekly_table, how='inner')
        data_table = self.scale_data_table(data_table)
        return data_table


    def scale_data_table(self, data_table):
        if self.scale_data == "scale":
            data_table[self.predictors_col_names] = scale(data_table[self.predictors_col_names])
        elif self.scale_data == "minmax":
            data_table[self.predictors_col_names] = minmax_scale(data_table[self.predictors_col_names])
        return data_table


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


if __name__ == "__main__":

    from skmultiflow.data import DataStream, RegressionGenerator
    target_label = "new_cases"
    no_hist_days = 0
    no_hist_weeks = 2
    begin_test_date = "2021-11-06"
    scale_data = None

    data = Data(
        no_hist_days=no_hist_days,
        no_hist_weeks=no_hist_weeks,
        target_label=target_label,
        begin_test_date=begin_test_date,
        scale_data=scale_data
    )

    X_train, y_train, X_test_t, y_test_t = data.get_data()
    stream = DataStream(X_test_t, y_test_t)
