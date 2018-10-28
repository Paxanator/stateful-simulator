

from abc import ABC, abstractmethod
from stateful_simulator.datatypes.DataTypes import FeatureVector
from typing import List

class StatelessModel(ABC):

    @abstractmethod
    def predict(self, fv: FeatureVector) -> float:
        pass

    @abstractmethod
    def train(self, fvs: List[FeatureVector]):
        pass