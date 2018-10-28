from stateful_simulator.metrics.Metrics import Metrics, PointMetric
from stateful_simulator.metrics.AgreementMeasure import AgreementMeasure
from stateful_simulator.metrics.LatenessMeasure import LatenessMeasure
from stateful_simulator.datatypes.DataTypes import Prediction

from typing import List

class DefaultMeasures(Metrics):

    def __init__(self, no_delay: List[Prediction], delay: List[Prediction]) -> None:
        self.measures = [AgreementMeasure(no_delay,delay), LatenessMeasure(no_delay,delay)]

    def point_metrics(self) -> List[PointMetric]:
        return {metric.name: metric.value for measure in self.measures for metric in measure.point_metrics()}
