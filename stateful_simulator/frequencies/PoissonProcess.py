

from stateful_simulator.frequencies.DataFrequency import DataFrequency
import math
from random import random
from datetime import datetime, timedelta


class PoissonProcess(DataFrequency):

    def __init__(self, inter_arrival_s: float)-> None:
        self.inter_arrival_s = inter_arrival_s

    def next_time(self, current_time: datetime) -> datetime:
        return current_time + timedelta(seconds=(-1*self.inter_arrival_s) * math.log(random()))

    @property
    def frequency(self):
        return 1/self.inter_arrival_s
