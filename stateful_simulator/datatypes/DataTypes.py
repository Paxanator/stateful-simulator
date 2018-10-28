from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Union, Generator
from dataclasses import dataclass
from operator import itemgetter

@dataclass
class RecordTime:
    event_time: datetime
    process_time: datetime

    def time_difference(self):
        return (self.process_time - self.event_time).total_seconds()

    def __lt__(self, other):
        return self.event_time < other.event_time

    def __le__(self, other):
        return self.event_time <= other.event_time

    def __gt__(self, other):
        return self.event_time > other.event_time

    def __ge__(self, other):
        return self.event_time >= other.event_time

    def drop(self,other: RecordTime)-> bool:
        return self.process_time < other.process_time

@dataclass
class Prediction:
    prediction: float
    target: Union[float, None]
    timestamp: RecordTime
    num_points: int

@dataclass
class FeatureVector:
    features: List[float]
    target: Union[float, None]
    timestamp: RecordTime
    num_points: int

@dataclass
class TimeSeriesChunk:
    timestamp: RecordTime
    target: float
    numeric_features: List[List[float]]

@dataclass
class TimeSeriesDataSet:
    timestamps: List[RecordTime]
    target: List[float]
    train_timestamp: RecordTime
    numeric_features: List[List[float]]
    feature_cols: List[str]
    sorted: bool = False

    def sort(self):
        if not self.sorted:
            indexes = [i[0] for i in sorted(enumerate(self.timestamps), key=lambda x: x[1])]
            self.timestamps = [self.timestamps[i] for i in indexes]
            self.target = [self.target[i] for i in indexes]
            self.numeric_features = [self.numeric_features[i] for i in indexes]

    def train_chunks(self, lookback: timedelta)-> Generator[TimeSeriesChunk,None,None]:
        self.sort()
        training_timestamps = [timestamp for timestamp in self.timestamps if timestamp < self.train_timestamp]
        for ix, training_timestamp in enumerate(training_timestamps):
            endx = ix
            ending_timstamp = RecordTime(training_timestamp.event_time - lookback, training_timestamp.process_time)
            temp_time = training_timestamp
            while endx > 0 and ending_timstamp < temp_time:
                endx = endx - 1
                temp_time = self.timestamps[endx]
            yield TimeSeriesChunk(training_timestamp, self.target[ix], self.numeric_features[endx:ix+1]) # exclusive


    def test_chunks(self, lookback: timedelta, drop_late_records: bool) -> Generator[TimeSeriesChunk,None,None]:
        self.sort()
        testing_timestamps = [timestamp for timestamp in self.timestamps if timestamp >= self.train_timestamp]
        offset = len(self.timestamps) - len(testing_timestamps)
        for ix, training_timestamp in enumerate(testing_timestamps):
            ix = ix + offset
            endx = ix
            ending_timstamp = RecordTime(training_timestamp.event_time - lookback, training_timestamp.process_time)
            temp_time = training_timestamp
            while endx > 0 and ending_timstamp < temp_time:
                endx = endx - 1
                temp_time = self.timestamps[endx]
            if not drop_late_records:
                num_features = self.numeric_features[endx:ix+1] #exclusive
            else:
                indices_to_keep = [index for index in range(endx, ix+1)
                                   if not training_timestamp.drop(self.timestamps[index])]
                if len(indices_to_keep) == 1:
                    num_features = [self.numeric_features[indices_to_keep[0]]]
                else:
                    num_features = list(itemgetter(*indices_to_keep)(self.numeric_features))


            yield TimeSeriesChunk(training_timestamp, self.target[ix], num_features)