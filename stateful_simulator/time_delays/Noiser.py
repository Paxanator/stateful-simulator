
from abc import ABC, abstractmethod

from stateful_simulator.datatypes.DataTypes import TimeSeriesDataSet, RecordTime
from datetime import datetime


class Noiser(ABC):

    def add_delay(self, dataset: TimeSeriesDataSet) ->  TimeSeriesDataSet:
        timestamps = dataset.timestamps
        new_timestamps = [RecordTime(timestamp.event_time, self.generate_process_time(timestamp.event_time))
                          for timestamp in timestamps]
        dataset.timestamps = new_timestamps
        return dataset

    @property
    @abstractmethod
    def noise_intensity(self):
        pass

    @abstractmethod
    def generate_process_time(self, event_time: datetime) -> datetime:
        pass

    def plot_delay(self,dataset: TimeSeriesDataSet, output_file: str):
        pass