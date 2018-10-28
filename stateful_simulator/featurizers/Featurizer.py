
from abc import ABC, abstractmethod

from stateful_simulator.datatypes.DataTypes import TimeSeriesChunk, FeatureVector

class Featurizer(ABC):

    @abstractmethod
    def featurize(self, df: TimeSeriesChunk) -> FeatureVector:
        pass
