from datetime import datetime
from faust import Record

from typing import *


class ScoredRecord(Record):
    id: str
    key: str

    event_time: datetime
    process_time: datetime
    time_delay: float

    fields: Dict[str,float]
    scores: Dict[str,float]

