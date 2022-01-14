import json
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def plot_mean_and_deviations(y_pred_avg, y_pred_std, y_test):

    plt.style.use('default')
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1, 1, 1)
    ax.title.set_fontsize(13)
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)
    ax.tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_ylim([-0.1, 0.7])
    ax.set_xlabel("Timestamps")
    ax.set_ylabel("New cases")

    x = list(range(1, len(y_test) + 1))
    def prepare_plot(means_arr, stds_arr, ax, color, label, marker):
        ax.plot(x, means_arr, color=color, alpha=0.9, lw=1, label=label, marker=marker)
        std_upper = np.minimum(means_arr + 3*stds_arr, 100000)
        std_lower = np.maximum(means_arr - 3*stds_arr, 0)
        ax.fill_between(x, std_lower, std_upper, color=color, alpha=0.2)

    prepare_plot(y_pred_avg, y_pred_std, ax, 'tab:orange', "Predictions", "+")
    ax.plot(x, y_test, color='tab:purple', label="Ground truth", marker="+")
    ax.legend(frameon=False)
    # fig.savefig(f"figures/err_rate{key}.png", dpi=300)
    plt.show()
