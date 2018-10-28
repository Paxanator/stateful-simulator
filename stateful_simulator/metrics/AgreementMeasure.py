from stateful_simulator.metrics.Metrics import Metrics, PointMetric
from stateful_simulator.datatypes.DataTypes import Prediction

from typing import List
from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy import array


class AgreementMeasure(Metrics):

    def point_metrics(self) -> List[PointMetric]:
        return [self.rmse_difference(),self.delay_rmse(),self.no_delay_rmse()]

    def rmse_difference(self) -> PointMetric:
        return PointMetric("rmse_difference",
                           "The difference in RMSE between delayed and non delayed dataset",
                           self.calc_rmse(self.delay) - self.calc_rmse(self.no_delay))

    def calc_rmse(self, preds: List[Prediction]):
        y_actual = array([pred.target for pred in preds])
        y_predicted = array([pred.prediction for pred in preds])
        return sqrt(mean_squared_error(y_actual, y_predicted))


    def no_delay_rmse(self):
        return PointMetric("rmse_no_delay",
                           "The RMSE of no delay",
                           self.calc_rmse(self.no_delay))

    def delay_rmse(self):
        return PointMetric("rmse_delay",
                           "The RMSE of delay",
                           self.calc_rmse(self.delay))