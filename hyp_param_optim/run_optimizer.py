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

    cfg_fname = "hyp_param_optim/configs/config_example.json"
    config = read_json(cfg_fname)
    config = ConfigParser(config)

    if config['server_address']:
        ray.init(f"ray://{config['server_address']}")
    else:
        ray.init(configure_logging=False, object_store_memory=78643200)
    main(config)






