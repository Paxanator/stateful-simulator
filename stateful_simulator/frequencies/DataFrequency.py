
from abc import ABC, abstractmethod
from datetime import timedelta, datetime


class DataFrequency(ABC):

    @abstractmethod
    def next_time(self, current_time: datetime) -> datetime:
        pass

    @property
    @abstractmethod
    def frequency(self):
        pass