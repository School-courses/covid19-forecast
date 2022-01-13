
from skmultiflow.meta import AdaptiveRandomForestRegressor
from skmultiflow.trees import (HoeffdingAdaptiveTreeRegressor,
                               HoeffdingTreeRegressor,
                               StackedSingleTargetHoeffdingTreeRegressor,
                               iSOUPTreeRegressor)

FORECAST_ALGO = {
    'ARF': AdaptiveRandomForestRegressor,
    'HT': HoeffdingTreeRegressor,
    'HAT': HoeffdingAdaptiveTreeRegressor,
    'STTHT': StackedSingleTargetHoeffdingTreeRegressor,
    'SOUPT': iSOUPTreeRegressor
}
