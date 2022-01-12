import os
import pickle
from abc import abstractmethod

import numpy as np
from ray import tune
from sklearn.utils import shuffle


class BaseOptimizer():
    def __init__(self, model, config):
        self.model = model
        self.tuned_params = self._modify_params(config)
        self.num_samples = config["num_samples"]
        self.save_dir = config.save_dir
        self.config = config

    def optimize(self):
        analysis = self.perform_search()
        train_report = self.create_train_report(analysis)
        self.save_report(train_report, "report_train.txt")
        self.save_report(str(analysis.best_result['mean_loss']), "loss_train.txt")

    def save_report(self, report, name_txt):
        save_path = os.path.join(self.save_dir, name_txt)
        with open(save_path, "w") as text_file:
            text_file.write(report)

    def create_train_report(self, analysis):
        '''Should return report from training'''
        return "Train report not configured."

    @staticmethod
    def _modify_params(config):
        tuned_parameters = config["tuned_parameters"]

        for method_name in tuned_parameters:
            temp = tuned_parameters[method_name]
            if temp[0] == 'CHOICE':
                temp.pop(0)
                tuned_parameters[method_name] = tune.choice(temp)
            elif len(temp) == 3 and temp[0] == 'LOG_UNIFORM':
                tuned_parameters[method_name] = tune.loguniform(temp[1], temp[2])
            elif len(temp) == 3 and temp[0] == 'RAND_INT':
                tuned_parameters[method_name] = tune.randint(temp[1], temp[2])
            else:
                raise Exception("Parameters not configured properly")

        return tuned_parameters

    @abstractmethod
    def perform_search(self):
        '''Seach of hyperparameters implemented here'''
        raise NotImplementedError

    @abstractmethod
    def objective(self, config):
        '''Should evaluate objective function used in perform search'''
        raise NotImplementedError


