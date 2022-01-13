import numpy as np
import ray
from ray import tune
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, mean_absolute_error,
                             mean_squared_error)

from hyp_param_optim.base_optimizer import BaseOptimizer


class OptimizerRegression(BaseOptimizer):
    def __init__(self, model, config):
        super().__init__(model, config)

    def objective(self, config):
        no_hist_vals = config["data_window_size"]
        config.pop("data_window_size")
        stream, train_data = self.create_stream(no_hist_vals)
        stream.restart()

        self.model.set_params(**config)
        self.model.fit(train_data[0], train_data[1])

        y_pred, y_test = [], []
        while stream.has_more_samples():
            x_t, y_t = stream.next_sample()
            y_p = self.model.predict(x_t)[0]
            self.model.partial_fit(x_t, y_t)
            y_pred.append(y_p)
            y_test.append(y_t)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        tune.report(mean_loss=rmse)



# class OptimizerRegression(BaseOptimizer):
#     def __init__(self, model, config):
#         super().__init__(model, config)

#     def objective(self, config):
#         no_hist_vals = config["data_window_size"]
#         config.pop("data_window_size")
#         stream = self.create_stream("new_cases", no_hist_vals)
#         stream.restart()

#         self.model.set_params(**config)

#         y_pred, y_test = [], []
#         while stream.has_more_samples():
#             x_t, y_t = stream.next_sample()
#             y_p = self.model.predict(x_t)[0]
#             self.model.partial_fit(x_t, y_t)
#             y_pred.append(y_p)
#             y_test.append(y_t)

#         rmse = mean_squared_error(y_test, y_pred, squared=False)
#         tune.report(mean_loss=rmse)
