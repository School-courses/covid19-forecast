import numpy as np
import pandas as pd


class Data():
    def __init__(self, no_hist_values, target_label):
        self.df_daily = pd.read_csv("data/slovenia_daily.csv")
        self.df_weekly = pd.read_csv("data/slovenia_weekly.csv")

        self.df_daily['date'] = pd.to_datetime(self.df_daily['date'])
        self.df_weekly['date'] = pd.to_datetime(self.df_weekly['date'])

        self.target_label = target_label
        self.no_hist_vals = no_hist_values
        self.predictors_col_names = None

        self.df_data_table = self.create_table()


    def create_table(self):
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
        self.predictors_col_names = col_names

        df_data_table = pd.concat([date_target_df, target_df, predictors_df], axis=1)
        return df_data_table


    def get_data(self, save=False):
        if save:
            self.df_data_table.to_csv("data/data_table.csv", index=False)
        X = self.df_data_table.loc[:, self.predictors_col_names].to_numpy()
        y = self.df_data_table.loc[:, "target"].to_numpy()
        return X, y


