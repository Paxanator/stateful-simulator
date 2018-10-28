

from abc import ABC
from datetime import timedelta

from stateful_simulator.datatypes.DataTypes import TimeSeriesDataSet,Prediction, TimeSeriesChunk, FeatureVector
from stateful_simulator.models.StatelessModel import StatelessModel
from stateful_simulator.featurizers.Featurizer import Featurizer
from typing import List

class StatefulModel(ABC):
    def __init__(self, model: StatelessModel, featurizer: Featurizer, lookback: timedelta)-> None:
        self.model = model
        self.lookback = lookback
        self.featurizer = featurizer

    def train(self, df: TimeSeriesDataSet):
        self.model.train([self.featurizer.featurize(chunk) for chunk in df.train_chunks(lookback=self.lookback)])

    def predict(self, df: TimeSeriesDataSet, drop_late_records: bool)-> List[Prediction]:
        return [self.process_chunk(chunk) for chunk in df.test_chunks(lookback=self.lookback, drop_late_records=drop_late_records)]

    def process_chunk(self, df: TimeSeriesChunk)-> Prediction:
        return self.predict_one(self.featurizer.featurize(df))

    def predict_one(self, features: FeatureVector)-> Prediction:
        return Prediction(self.model.predict(features),
                          features.target,
                          features.timestamp,
                          features.num_points)