
from stator.records.DataRecord import DataRecord
from abc import ABC
from typing import *


class StatefulModel(ABC):
    def __init__(self, name):
        self.name = name

    def score(self, records: List[DataRecord]) -> float:
        pass

    def train(self, records: List[List[DataRecord]]):
        pass

    @property
    def name(self):
        self.name
