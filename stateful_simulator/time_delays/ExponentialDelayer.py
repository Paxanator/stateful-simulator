

from stateful_simulator.time_delays.Delayer import Delayer
import math
from random import random
from datetime import datetime,timedelta


class ExponentialDelayer(Delayer):

    def __init__(self, mean_delay_s: float)-> None:
        self.mean_delay_s = mean_delay_s

    def generate_process_time(self, event_time: datetime) -> datetime:
        return event_time + timedelta(seconds=(-1*self.mean_delay_s) * math.log(random()))

    @property
    def delay_intensity(self):
        return self.mean_delay_s
