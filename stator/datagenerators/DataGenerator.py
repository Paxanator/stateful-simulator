
from abc import ABC
from typing import List

from stator.records.DataRecord import DataRecord


class DataGenerator(ABC):

    def get_batch(self) -> List[DataRecord]:
        pass

    def get_training_data(self) -> List[DataRecord]:
        pass


