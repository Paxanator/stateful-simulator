
from stator.datagenerators.DataGenerator import DataGenerator
from stator.configs.KafkaConfig import KafkaConfig
from stator.records.DataRecord import DataRecord

from time import sleep

from faust.app import App
from typing import List, Callable

class DataEmitter:
    def __init__(self, data_generator: DataGenerator, kafka_config: KafkaConfig, topic: str, app_name: str = "Generator") -> None:
        self.data_generator = data_generator
        self.app = App(app_name, broker=KafkaConfig.brokers, store=kafka_config.store)
        self.kafka_config = kafka_config
        self.topic = self.app.topic(topic, key_type=str, value_type=DataRecord)

    def send_forever(self, interval_generator: Callable):
        record_batch = [] # type: List[DataRecord]
        while True:
            if len(record_batch) == 0:
                record_batch = self.data_generator.get_batch()
            self.send(record_batch.pop(0))
            sleep(interval_generator())

    def send(self, record: DataRecord):
        pass