import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict


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


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
