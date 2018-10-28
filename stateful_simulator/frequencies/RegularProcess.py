

from stateful_simulator.frequencies.DataFrequency import DataFrequency
from random import random
from datetime import datetime, timedelta


class RegularProcess(DataFrequency):

    def __init__(self, lower_inter_arrival_s: float, upper_inter_arrival_s: float)-> None:
        self.lower_inter_arrival_s = lower_inter_arrival_s
        self.upper_inter_arrival_s = upper_inter_arrival_s

    def next_time(self, current_time: datetime) -> datetime:
        return current_time + timedelta(seconds= (self.upper_inter_arrival_s-self.lower_inter_arrival_s) * random()
                                                 + self.lower_inter_arrival_s)

    @property
    def frequency(self):
        return 2 / (self.lower_inter_arrival_s + self.upper_inter_arrival_s)