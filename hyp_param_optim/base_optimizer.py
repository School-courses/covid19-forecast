import os
import pickle
from abc import abstractmethod

import numpy as np
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler, HyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, mean_absolute_error,
                             mean_squared_error)
from skmultiflow.data import DataStream

from data_management.data import Data


class BaseOptimizer():
    def __init__(self, model, config):
        self.model = model
        self.tuned_params = self._modify_params(config)
        self.num_samples = config["num_samples"]
        self.root_dir = config["root_dir"]
        self.selected_features_pkl = config["selected_features_pkl_name"]
        self.save_dir = config.save_dir
        self.target_label = "new_cases"
        self.begin_test_date = "2021-11-06"
        self.best_config = 0
        self.config = config
        self.feats_names_selected = None

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
        train_report = "Best_results:\n"
        train_report += f"{analysis.best_result}\n\n"
        train_report += f"Best hyperparameters found were:\n"
        train_report += f"{analysis.best_config}\n\n"
        train_report += f"Best score:\n"
        train_report += f"{analysis.best_result['mean_loss']}"
        print(train_report)
        return train_report

    @staticmethod
    def _modify_params(config):
        tuned_parameters = config["tuned_parameters"]

        for method_name in tuned_parameters:
            temp = tuned_parameters[method_name]
            if temp[0] == 'CHOICE':
                temp.pop(0)
                tuned_parameters[method_name] = tune.choice(temp)
            elif temp[0] == 'GRID':
                temp.pop(0)
                tuned_parameters[method_name] = tune.grid_search(temp)
            elif len(temp) == 3 and temp[0] == 'LOG_UNIFORM':
                tuned_parameters[method_name] = tune.loguniform(temp[1], temp[2])
            elif len(temp) == 3 and temp[0] == 'RAND_INT':
                tuned_parameters[method_name] = tune.randint(temp[1], temp[2])
            else:
                raise Exception("Parameters not configured properly")

        return tuned_parameters

    def perform_search(self):
        # algo = HyperOptSearch()
        # algo = ConcurrencyLimiter(algo, max_concurrent=10)

        scheduler = HyperBandScheduler()
        analysis = tune.run(
            self.objective,
            name="covid_baby",
            # search_alg=algo,
            scheduler=scheduler,
            resources_per_trial={"cpu":1, "gpu":0.1},
            metric="mean_loss",
            mode="min",
            keep_checkpoints_num=5,
            num_samples=self.num_samples,
            config=self.tuned_params
            )

        self.best_config = analysis.best_config
        return analysis

    def create_stream(self, no_hist_days, no_hist_weeks, scale_data):
        data = Data(
            no_hist_days=no_hist_days,
            no_hist_weeks=no_hist_weeks,
            target_label=self.target_label,
            root_dir=self.root_dir,
            begin_test_date=self.begin_test_date,
            scale_data=scale_data
        )
        if self.selected_features_pkl != "" and self.feats_names_selected is None:
            with open(os.path.join(self.root_dir, "output/features", self.selected_features_pkl), "rb") as file:
                self.feats_names_selected = pickle.load(file)
            with open(os.path.join(self.save_dir, self.selected_features_pkl), "wb") as file:
                pickle.dump(self.feats_names_selected, file)

        if self.feats_names_selected is not None:
            data.predictors_col_names = self.feats_names_selected

        X_train, y_train, X_test, y_test = data.get_data()
        return DataStream(X_test, y_test), (X_train, y_train)


    @abstractmethod
    def objective(self, config):
        '''Should evaluate objective function used in perform search'''
        raise NotImplementedError


