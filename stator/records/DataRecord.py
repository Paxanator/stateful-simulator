from datetime import datetime
from faust import Record

from typing import *


class DataRecord(Record):
    event_time: datetime
    time_delay: float
    id: str
    key: str
    fields: Dict[str,float]

