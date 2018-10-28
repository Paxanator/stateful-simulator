

from stateful_simulator.time_delays.Noiser import Noiser
import math
from random import random
from datetime import datetime,timedelta


class ExponentialNoiser(Noiser):

    def __init__(self, mean_arrival_s: float)-> None:
        self.mean_arrival_s = mean_arrival_s

    def generate_process_time(self, event_time: datetime) -> datetime:
        return event_time + timedelta(seconds=(-1*self.mean_arrival_s) * math.log(random()))

    @property
    def noise_intensity(self):
        return self.mean_arrival_s
