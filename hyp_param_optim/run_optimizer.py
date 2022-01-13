import argparse
import collections
import glob
import os

import ray

import hyp_param_optim.models as models_
import hyp_param_optim.optimizers as optimizers_
from hyp_param_optim.parse_config import ConfigParser
from utils.utils import read_json


def main(config):

    model = config.init_obj('model', models_).created_model()
    Optimizer = config.import_module('optimizer', optimizers_)

    optim = Optimizer(model=model,
                      config=config)

    optim.optimize()


if __name__ == '__main__':


    config_list = glob.glob("hyp_param_optim/configs/*.json")
    config_list = [config_list[0]]

    for cfg_fname in config_list:
        config = read_json(cfg_fname)
        config["num_samples"] = 1000
        config = ConfigParser(config)
        main(config)






