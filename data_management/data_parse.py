import pandas as pd
import utils.utils as utils

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

# Restart indexes that does not represent whole week
start_idx = utils.get_first_row_with(lambda x: x.day == 7, df_daily)
end_idx = utils.get_last_row_with(lambda x: x.day == 6, df_daily)
df_daily = df_daily.iloc[start_idx:end_idx+1].reset_index(drop=True)

# Get weekly sums and add week of the year
df_weekly = df_daily.copy()
df_weekly = df_weekly.resample('W-Sat', on='date').sum().reset_index().sort_values(by='date')
df_weekly['week'] = df_weekly['date'].dt.isocalendar().week

# Output formatting
df_weekly = df_weekly[['date', 'week','value_x', 'value_y']]
df_weekly.columns = ['date', 'week', 'new_cases', 'new_deaths']
df_daily = df_daily[['date', 'day', 'week', 'month','value_x', 'value_y']]
df_daily.columns = ['date', 'day', 'week', 'month', 'new_cases', 'new_deaths']

# Save to CSV
df_weekly.to_csv("data/slovenia_weekly.csv", index=False)
df_daily.to_csv("data/slovenia_daily.csv", index=False)



