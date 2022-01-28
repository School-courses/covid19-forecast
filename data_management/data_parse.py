from functools import reduce

import pandas as pd
from meteostat import Daily, Point

import utils.utils as utils


def main():
    ### LOAD NUMBER OF CASES AND DEATHS ###
    #######################################
    # Load the data
    url_new_cases = "https://raw.githubusercontent.com/covid19-forecast-hub-europe/covid19-forecast-hub-europe/main/data-truth/JHU/truth_JHU-Incident%20Cases.csv"
    url_new_deaths = "https://raw.githubusercontent.com/covid19-forecast-hub-europe/covid19-forecast-hub-europe/main/data-truth/JHU/truth_JHU-Incident%20Deaths.csv"

    df_deaths = pd.read_csv(url_new_deaths)
    df_cases = pd.read_csv(url_new_cases)

    # Merge into a single dataframe
    df_all = df_cases.merge(df_deaths, how='inner', on=["location_name", "date"])

    # Extract data for Slovenia only
    df_slo = df_all[df_all['location_name'] == 'Slovenia'].reset_index(drop=True)
    df_slo['date'] = pd.to_datetime(df_slo['date'])

    # Get daily cases
    df_daily = df_slo.copy()
    df_daily['day'] = df_daily['date'].dt.isocalendar().day
    df_daily['week'] = df_daily['date'].dt.isocalendar().week
    df_daily['month'] = df_daily['date'].dt.month


    ### LOAD OTHER RELEVANT DATA ###
    ################################
    # How many tests were performed
    url_tests_performed = "https://raw.githubusercontent.com/sledilnik/data/master/csv/lab-tests.csv"
    df_tests = pd.read_csv(url_tests_performed)[["date", "tests.performed", "tests.positive"]].fillna(0)
    # Number of active cases
    url_region_active = "https://raw.githubusercontent.com/sledilnik/data/master/csv/region-active.csv"
    df_region_active = pd.read_csv(url_region_active).fillna(0)
    # Number of patients
    url_patients = "https://raw.githubusercontent.com/sledilnik/data/master/csv/patients.csv"
    df_patients = pd.read_csv(url_patients)[[
        "date", 'state.in_hospital', 'state.icu', 'state.critical', 'state.care', 'state.deceased'
        ]].fillna(0)
    # School cases
    url_school = "https://raw.githubusercontent.com/sledilnik/data/master/csv/schools-cases.csv"
    df_school = pd.read_csv(url_school)[[
        "date", 'kindergarten.employees.confirmed', 'kindergarten.attendees.confirmed', 'elementary.employees.confirmed',
        'elementary.attendees.confirmed', 'highschool.employees.confirmed', 'highschool.attendees.confirmed',
        'institutions.employees.confirmed', 'institutions.attendees.confirmed'
        ]].fillna(0)
    # Weather data
    weather_start = df_daily['date'].min()
    weather_end = df_daily['date'].max()
    ljubljana = Point(46.056946, 14.505751, 70)
    data_weather = Daily(ljubljana, weather_start, weather_end)
    df_weather = data_weather.fetch().reset_index().fillna(0)
    df_weather = df_weather.rename(columns={"time":"date"})
    df_weather = df_weather.drop(columns=["tsun", "wdir"])
    df_weather["date"] = df_weather["date"].dt.strftime('%Y-%m-%d')

    # Merge all together
    dfs = [df_tests, df_region_active, df_patients, df_school, df_weather]
    df_other_info = reduce(lambda left, right: pd.merge(left, right, how="inner", on="date"), dfs)
    df_other_info['date'] = pd.to_datetime(df_other_info['date'])
    df_daily = pd.merge(df_daily, df_other_info, how="inner")

    # Restart indexes that does not represent whole week
    start_idx = utils.get_first_row_with(lambda x: x.day == 7, df_daily)
    end_idx = utils.get_last_row_with(lambda x: x.day == 6, df_daily)
    df_daily = df_daily.iloc[start_idx:end_idx+1].reset_index(drop=True)


    ### SAVE MERGED CSV ###
    #######################
    # Get weekly sums and add week of the year
    df_weekly = df_daily.copy()
    df_weekly = df_weekly.resample('W-Sat', on='date').sum().reset_index().sort_values(by='date')
    df_weekly['week'] = df_weekly['date'].dt.isocalendar().week

    # Output formatting
    col_names = df_daily.columns.to_list()
    names_to_exclude = ['date', 'day', 'week', 'month','value_x', 'value_y', 'location_x', 'location_name', 'location_y']
    col_add_names = [name for name in col_names if name not in names_to_exclude]
    col_add_names_ = [name.replace('.', '_') for name in col_names if name not in names_to_exclude]

    df_daily = df_daily[['date', 'day', 'week', 'month','value_x', 'value_y'] + col_add_names]
    df_daily.columns = ['date', 'day', 'week', 'month', 'new_cases', 'new_deaths'] + col_add_names_
    df_weekly = df_weekly[['date', 'week','value_x', 'value_y'] + col_add_names]
    df_weekly.columns = ['date', 'week', 'new_cases', 'new_deaths'] + col_add_names_

    # Save to CSV
    df_daily.to_csv("data/slovenia_daily.csv", index=False)
    df_weekly.to_csv("data/slovenia_weekly.csv", index=False)


if __name__ == "__main__":
    main()
