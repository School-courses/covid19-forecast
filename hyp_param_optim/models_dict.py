
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from skmultiflow.meta import AdaptiveRandomForestRegressor
from skmultiflow.trees import (HoeffdingAdaptiveTreeRegressor,
                               HoeffdingTreeRegressor,
                               StackedSingleTargetHoeffdingTreeRegressor,
                               iSOUPTreeRegressor)

from utils.helpers import MultiflowPredictorWrapper

multiflow_algo = {
    'ARF': AdaptiveRandomForestRegressor(random_state=1),
    'HT': HoeffdingTreeRegressor,
    'HAT': HoeffdingAdaptiveTreeRegressor,
    'STTHT': StackedSingleTargetHoeffdingTreeRegressor,
    'SOUPT': iSOUPTreeRegressor
}

sklearn_algo = {
    'SVR': MultiflowPredictorWrapper(SVR()),
    'RF': MultiflowPredictorWrapper(RandomForestRegressor())
}

FORECAST_ALGO = {**multiflow_algo, **sklearn_algo}


