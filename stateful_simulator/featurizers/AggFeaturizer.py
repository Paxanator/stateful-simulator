from abc import ABC, abstractmethod

from stateful_simulator.featurizers.Featurizer import Featurizer
from stateful_simulator.datatypes.DataTypes import TimeSeriesChunk, FeatureVector
from typing import List, Callable


class AggFeaturizer(Featurizer):

    def __init__(self, op: Callable[[List[float]], float]) -> None:
        self.op = op

    def featurize(self, df: TimeSeriesChunk) -> FeatureVector:
        transposed_features = list(map(list, zip(*df.numeric_features)))  # type: List[List[float]]
        return FeatureVector(list(map(self.op, transposed_features)),
                             df.target,
                             df.timestamp,
                             len(df.numeric_features))
