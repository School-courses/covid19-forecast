
from skmultiflow.lazy import KNNRegressor
from skmultiflow.meta import AdaptiveRandomForestRegressor
from skmultiflow.trees import (HoeffdingTreeRegressor,
                               StackedSingleTargetHoeffdingTreeRegressor,
                               iSOUPTreeRegressor)

methods_dict = {
    'KNN': KNNRegressor,
    'ARF': AdaptiveRandomForestRegressor,
    'HT': HoeffdingTreeRegressor,
    'STT': StackedSingleTargetHoeffdingTreeRegressor,
    'SOUPTR': iSOUPTreeRegressor
}
