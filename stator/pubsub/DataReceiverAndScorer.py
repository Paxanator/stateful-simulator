
from faust.app import App

from stator.configs.KafkaConfig import KafkaConfig
from stator.models.StatefulModel import StatefulModel
from stator.records.DataRecord import DataRecord
from stator.records.ScoredRecord import ScoredRecord

from typing import *

class DataReceiverAndScorer:
    def __init__(self, kafka_config: KafkaConfig, input_topic: str, output_topic: str, app_name: str = "Scorer") -> None:
        self.app = App(app_name, broker=KafkaConfig.brokers, store=kafka_config.store)
        self.models = []  # type: List[StatefulModel]
        self.input_topic = self.app.topic(input_topic, key_type=str, value_type=DataRecord)
        self.output_topic = self.app.topic(output_topic, key_type=str, value_type=ScoredRecord)

        self.state_store = self.app.Table(name="state_store", default=list, key_type=str, value_type=List[DataRecord])

    def register_model(self, model: StatefulModel):
        self.models.append(model)

    def score_forever(self):
        scoring_function = self.app.agent(self.score_one)


    def score_one(self, records: List[DataRecord]):
        for model in self.models:
            yield {model.name: model.score(records)}
