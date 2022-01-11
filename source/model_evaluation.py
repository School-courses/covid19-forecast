import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skmultiflow.data import DataStream
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.lazy import KNNRegressor
from skmultiflow.meta import AdaptiveRandomForestRegressor, RegressorChain
from skmultiflow.trees import (HoeffdingTreeRegressor,
                               StackedSingleTargetHoeffdingTreeRegressor,
                               iSOUPTreeRegressor)

import utils.utils as utils
from data_management.data import Data

target_label = "new_cases"
no_hist_vals = 4

data = Data(no_hist_vals, target_label)
X, y = data.get_data()

stream = DataStream(X, y)
reg1 = AdaptiveRandomForestRegressor(random_state=1)
reg2 = KNNRegressor()
reg3 = HoeffdingTreeRegressor()

evaluator = EvaluatePrequential(max_samples = 1000000,
                                n_wait = 1,
                                pretrain_size = 20,
                                output_file = "output/eval_report.txt",
                                show_plot = True,
                                metrics = ['mean_absolute_error'])
# Run evaluation
evaluator.evaluate(stream = stream,
                   model = [reg1, reg2, reg3],
                   model_names = ['reg1', 'reg2', 'reg3']
)
print(evaluator.evaluation_summary())

