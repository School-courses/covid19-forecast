import argparse
import collections
import glob
import os

import ray
import sklearn.model_selection as model_selection_
from skmultiflow.data import DataStream

import hyp_param_optim.models as models_
import hyp_param_optim.optimizers as optimizers_
from data_management.data import Data
from hyp_param_optim.parse_config import ConfigParser
from utils.utils import read_json


def main(config):

    target_label = "new_cases"
    no_hist_vals = 4
    data = Data(no_hist_vals, target_label)
    X, y = data.get_data()
    stream = DataStream(X, y)

    model = config.init_obj('model', models_).created_model()
    Optimizer = config.import_module('optimizer', optimizers_)

    optim = Optimizer(model=model,
                      stream=stream,
                      config=config)

    optim.optimize()


if __name__ == '__main__':

    cfg_fname = "hyp_param_optim/configs/config_example.json"
    config = read_json(cfg_fname)
    config = ConfigParser(config)

    if config['server_address']:
        ray.init(f"ray://{config['server_address']}")
    else:
        ray.init(configure_logging=False, object_store_memory=78643200)
    main(config)






