

from stateful_simulator.time_delays.Delayer import Delayer
from random import random
from datetime import datetime,timedelta


class UniformDelayer(Delayer):

    def __init__(self, lower_s: float, upper_s: float)-> None:
        self.lower_s = lower_s
        self.upper_s = upper_s

    def generate_process_time(self, event_time: datetime) -> datetime:
        return event_time + timedelta(seconds= (self.upper_s - self.lower_s) * random() + self.lower_s)

    @property
    def delay_intensity(self):
        return (self.upper_s + self.lower_s) / 2