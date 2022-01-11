import matplotlib.pyplot as plt


def plot_data(title="label", *data_list):
    # plt.figure()
    plt.title(title)
    [plt.plot(data) for data in data_list]


def get_first_row_with(condition, df):
    """
    utils.get_first_row_with(lambda x: x.day == 6, df_test)
    """
    for index, row in df.iterrows():
        if condition(row):
            return index
    return None # Condition not met on any row in entire DataFrame


def get_last_row_with(condition, df):
    """
    utils.get_last_row_with(lambda x: x.day == 6, df_test)
    """
    df = df.iloc[::-1]
    for index, row in df.iterrows():
        if condition(row):
            return index
    return None # Condition not met on any row in entire DataFrame
