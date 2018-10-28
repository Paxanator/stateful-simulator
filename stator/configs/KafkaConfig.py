
from attr import dataclass
from typing import List


@dataclass
class KafkaConfig:
    name: str
    host: str
    store: str
    brokers: List[str]
