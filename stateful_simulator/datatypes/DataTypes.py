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

    def drop(self, other: RecordTime) -> bool:
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


# TODO: Misuse of dataclass, refactor

@dataclass
class TimeSeriesDataSet:
    timestamps: List[RecordTime]
    target: List[float]
    train_timestamp: RecordTime
    numeric_features: List[List[float]]
    feature_cols: List[str]
    start_time: RecordTime = None
    prepped: bool = False
    MIN_NUMBER_OF_TRAINING_POINTS: int = 100

    def prep_data(self):
        if not self.prepped:
            indexes = [i[0] for i in sorted(enumerate(self.timestamps), key=lambda x: x[1])]
            self.timestamps = [self.timestamps[i] for i in indexes]
            self.target = [self.target[i] for i in indexes]
            self.numeric_features = [self.numeric_features[i] for i in indexes]
            self.start_time = self.timestamps[0]

    def train_chunks(self, lookback: timedelta) -> Generator[TimeSeriesChunk, None, None]:
        self.prep_data()
        training_timestamps = [timestamp for timestamp in self.timestamps if timestamp < self.train_timestamp]

        for ix, training_timestamp in enumerate(training_timestamps):
            current_index = ix
            min_timestamp = RecordTime(training_timestamp.event_time - lookback,
                                       training_timestamp.process_time)
            last_index = self._get_ending_index(min_timestamp, training_timestamp, current_index)

            yield TimeSeriesChunk(training_timestamp,
                                  self.target[ix],
                                  self.numeric_features[last_index:current_index + 1])  # Include trigger

    def _get_ending_index(self, min_timestamp, current_time, current_index):
        last_index = current_index
        last_time = current_time
        while last_index > 0 and last_time > min_timestamp:
            last_index = last_index - 1
            last_time = self.timestamps[last_index]
        return last_index

    def test_chunks(self, lookback: timedelta, drop_late_records: bool) -> Generator[TimeSeriesChunk, None, None]:
        self.prep_data()
        # Assumed that lookback is valid based on train_chunks
        testing_timestamps = [timestamp for timestamp in self.timestamps if timestamp >= self.train_timestamp]
        offset = len(self.timestamps) - len(testing_timestamps)
        for ix, testing_timestamp in enumerate(testing_timestamps):
            current_index = ix + offset
            min_timestamp = RecordTime(testing_timestamp.event_time - lookback, testing_timestamp.process_time)
            last_index = self._get_ending_index(min_timestamp,testing_timestamp,current_index)
            if not drop_late_records:
                num_features = self.numeric_features[last_index: current_index + 1]  # Exclude last timestamp and include trigger
            else:
                indices_to_keep = [index for index in range(last_index, current_index + 1)
                                   if not testing_timestamp.drop(self.timestamps[index])]
                if len(indices_to_keep) == 1:
                    num_features = [self.numeric_features[indices_to_keep[0]]]
                else:
                    num_features = list(itemgetter(*indices_to_keep)(self.numeric_features))
            yield TimeSeriesChunk(testing_timestamp, self.target[current_index], num_features)
