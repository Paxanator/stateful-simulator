from stateful_simulator.datatypes.DataTypes import TimeSeriesDataSet, RecordTime
from stateful_simulator.featurizers.Featurizer import Featurizer
from stateful_simulator.frequencies.DataFrequency import DataFrequency

import pandas as pd
from typing import List, Callable
from datetime import timedelta, datetime
from scipy.stats import norm


def get_data_set(dataset_url: str,
                 timestamp_col: str,
                 timestamp_format: str,
                 target_col: str,
                 feature_cols: List[str],
                 train_percetange: float) -> TimeSeriesDataSet:
    df = pd.read_csv(dataset_url)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=timestamp_format)
    df.sort_values(by=timestamp_col)
    timestamps = [RecordTime(datetime, datetime) for datetime in df[timestamp_col]]
    targets = df[target_col]
    numeric_features = [list(record)[1:] for record in df[feature_cols].to_records()]
    train_timestamp = timestamps[round(len(timestamps) * train_percetange)]
    return TimeSeriesDataSet(timestamps, targets, train_timestamp, numeric_features, feature_cols, True)


def deterministic_dataset(decision_function: Callable[[List[float]], float],
                          random_error: Callable[[], float],
                          featurizer: Featurizer,
                          lookback: timedelta,
                          num_features: int,
                          num_points: int,
                          train_percentage: float,
                          frequency: DataFrequency) -> TimeSeriesDataSet:
    # Create brownian motion
    norms = norm.rvs(size=(num_points, num_features))

    init_time = datetime.now()
    current_time = init_time
    timestamps = []
    numeric_features = []
    targets = []
    for row in norms:
        timestamps.append(RecordTime(current_time,current_time))
        current_time = frequency.next_time(current_time)
        numeric_features.append(row.tolist())
        targets.append(0)  # We will redefine later

    train_timestamp = timestamps[round(len(timestamps) * train_percentage)]
    data_set = TimeSeriesDataSet(timestamps=timestamps,
                                 target=targets,
                                 train_timestamp=RecordTime(init_time, init_time),
                                 numeric_features=numeric_features,
                                 feature_cols=["feature_" + str(feature) for feature in range(num_features)],
                                 sorted=False)

    def predict_chunk(x):
        return decision_function(featurizer.featurize(x).features) + random_error()

    targets = [predict_chunk(chunk) for chunk in data_set.test_chunks(lookback=lookback, drop_late_records=False)]
    data_set.target = targets
    data_set.train_timestamp = train_timestamp
    return data_set
