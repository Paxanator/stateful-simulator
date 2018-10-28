

from abc import ABC, abstractmethod
from stateful_simulator.datatypes.DataTypes import Prediction

from dataclasses import dataclass
from typing import List


@dataclass
class PointMetric:
    name: str
    description: str
    value: float


class Metrics(ABC):
    def __init__(self, no_delay: List[Prediction], delay: List[Prediction]) -> None:
        self.no_delay = no_delay
        self.delay = delay

    @abstractmethod
    def point_metrics(self) -> List[PointMetric]:
        pass
