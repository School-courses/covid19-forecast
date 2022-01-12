import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler, HyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, mean_absolute_error,
                             mean_squared_error)
from sklearn.model_selection import cross_val_score
from skmultiflow.data import DataStream

from data_management.data import Data
from hyp_param_optim.base_optimizer import BaseOptimizer


class OptimizerRegression(BaseOptimizer):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.best_config = 0
        self.root_dir = config["root_dir"]

    def perform_search(self):
        algo = HyperOptSearch()
        algo = ConcurrencyLimiter(algo, max_concurrent=10)

        scheduler = HyperBandScheduler()
        analysis = tune.run(
            self.objective,
            name="covid_baby",
            search_alg=algo,
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

    def create_stream(self, target_label, no_hist_vals):
        target_label = "new_cases"
        data = Data(no_hist_vals, target_label, self.root_dir)
        X, y = data.get_data()
        return DataStream(X, y)

    def objective(self, config):
        no_hist_vals = config["data_window_size"]
        config.pop("data_window_size")
        stream = self.create_stream("new_cases", no_hist_vals)
        stream.restart()

        self.model.set_params(**config)

        y_pred, y_test = [], []
        while stream.has_more_samples():
            x_t, y_t = stream.next_sample()
            y_p = self.model.predict(x_t)[0]
            self.model.partial_fit(x_t, y_t)
            y_pred.append(y_p)
            y_test.append(y_t)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        tune.report(mean_loss=rmse)

    def create_train_report(self, analysis):
        train_report = "Best_results:\n"
        train_report += f"{analysis.best_result}\n\n"
        train_report += f"Best hyperparameters found were:\n"
        train_report += f"{analysis.best_config}\n\n"
        train_report += f"Best score:\n"
        train_report += f"{analysis.best_result['mean_loss']}"
        print(train_report)
        return train_report


