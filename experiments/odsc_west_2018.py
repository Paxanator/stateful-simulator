from stateful_simulator.models.SklearnModel import SklearnModel
from stateful_simulator.models.StatefulModel import StatefulModel
from stateful_simulator.time_delays.ExponentialDelayer import ExponentialDelayer
from stateful_simulator.time_delays.UniformDelayer import UniformDelayer
from stateful_simulator.time_delays.Delayer import Delayer
from stateful_simulator.frequencies.DataFrequency import DataFrequency
from stateful_simulator.frequencies.PoissonProcess import PoissonProcess
from stateful_simulator.frequencies.RegularProcess import RegularProcess
from stateful_simulator.data_generators.util_data import deterministic_dataset
from stateful_simulator.featurizers.AggFeaturizer import AggFeaturizer
from stateful_simulator.metrics.DefaultMeasures import DefaultMeasures

from sklearn.linear_model import LinearRegression
from datetime import timedelta
from random import random
from typing import Dict, List
from datetime import datetime

import pandas as pd
import os


def run_one(lookback: int
            , frequency: DataFrequency
            , delayer: Delayer
            , sensitive: bool) -> Dict[str, float]:
    if sensitive:
        aggregator = AggFeaturizer(lambda x: max(x))
    else:
        aggregator = AggFeaturizer(lambda x: sum(x) / len(x))
    state = timedelta(seconds=lookback)
    dataset = deterministic_dataset(
        decision_function=lambda y: sum([(index+1) * value for index, value in enumerate(y)]) + 4,
        random_error=random,
        featurizer=aggregator,
        lookback=state,
        num_features=5,
        train_percentage=.5,
        num_points=1000,
        frequency=frequency
    )

    delayed_dataset = delayer.add_delay(dataset)
    model = SklearnModel(LinearRegression())
    stateful_model = StatefulModel(model, aggregator, state)
    stateful_model.train(delayed_dataset)
    no_delay = stateful_model.predict(delayed_dataset, False)
    delay_preds = stateful_model.predict(delayed_dataset, True)
    measure = DefaultMeasures(no_delay, delay_preds)
    result = {"lookback_s": lookback,
              "delay_type": type(delayer).__name__,
              "delay_intensity_s": delayer.delay_intensity,
              "frequency_type": type(frequency).__name__,
              "frequency_s": frequency.frequency,
              "periodicity_s": 1/frequency.frequency,
              "sensitive": sensitive
              }
    result.update(measure.point_metrics())
    return (result)


def experiment(name
               , lookback_range: List[int] = [100]
               , delayers: List[Delayer] = [ExponentialDelayer(10)]
               , frequencies: List[DataFrequency] = [PoissonProcess(10)]
               , sensitivities: List[bool] = [False]):
    metrics = []
    print(f"Running experiment {name}")
    for lookback in lookback_range:
        for delayer in delayers:
            for frequency in frequencies:
                for sensitivity in sensitivities:
                    metrics.append(run_one(lookback, frequency, delayer, sensitivity))
    df = pd.DataFrame.from_records(metrics)
    df.to_csv(name + ".csv", index=False)
    return df


def main():
    experiment_name = "odsc_west_2018_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.mkdir(experiment_name)
    os.chdir(experiment_name)  # Sorry about changing dirs

    numeric_range = [1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200]

    def make_range_object(object, numeric_range):
        return [object(i) for i in numeric_range]

    lookback_range = make_range_object(lambda x: x, numeric_range)
    lookback_df = experiment("lookback", lookback_range=lookback_range)

    sensitivities = [True, False]
    sensitivity_df = experiment("sensitivities", lookback_range=lookback_range, sensitivities=sensitivities)

    delayers = make_range_object(ExponentialDelayer,numeric_range)
    noise_df = experiment("delay", delayers=delayers)

    frequencies = make_range_object(PoissonProcess, numeric_range)
    freq_df = experiment("frequencies", frequencies=frequencies)

    # Show Frequency and lookback are same
    experiment("freq_vs_lookback",
               frequencies=make_range_object(PoissonProcess, range(1,1000,10)),
               lookback_range=make_range_object(lambda x: x, range(1,1000,10)),
               delayers=[ExponentialDelayer(100)])

    # Interaction Effects
    interaction = experiment("interaction",
                             frequencies=frequencies,
                             delayers=delayers,
                             sensitivities=sensitivities)

    # Uniform Effects
    uniform_freqs = make_range_object(lambda x: RegularProcess(0,x),numeric_range)
    uniform_delays = make_range_object(lambda x: UniformDelayer(0,x),numeric_range)
    interaction = experiment("interaction_uniform",
                             frequencies=uniform_freqs,
                             delayers=uniform_delays,
                             sensitivities=sensitivities)

    os.chdir("..")


if __name__ == '__main__':
    main()
