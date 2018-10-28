from stateful_simulator.models.SklearnModel import SklearnModel
from stateful_simulator.models.StatefulModel import StatefulModel
from stateful_simulator.time_delays.ExponentialDelayer import ExponentialDelayer
from stateful_simulator.time_delays.Delayer import Delayer
from stateful_simulator.frequencies.DataFrequency import DataFrequency
from stateful_simulator.frequencies.PoissonProcess import PoissonProcess
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
            , noise_intensity: int
            , noise_type: Delayer
            , sensitive: bool) -> Dict[str, float]:
    if sensitive:
        aggregator = AggFeaturizer(lambda x: max(x))
    else:
        aggregator = AggFeaturizer(lambda x: sum(x) / len(x))
    state = timedelta(seconds=lookback)
    dataset = deterministic_dataset(
        decision_function=lambda y: sum([index * value for index, value in enumerate(y)]) + 4,
        random_error=random,
        featurizer=aggregator,
        lookback=state,
        num_features=5,
        train_percentage=.5,
        num_points=1000,
        frequency=frequency
    )
    noiser = noise_type(noise_intensity)
    delayed_dataset = noiser.add_delay(dataset)
    model = SklearnModel(LinearRegression())
    stateful_model = StatefulModel(model, aggregator, state)
    stateful_model.train(delayed_dataset)
    no_delay = stateful_model.predict(delayed_dataset, False)
    delay_preds = stateful_model.predict(delayed_dataset, True)
    measure = DefaultMeasures(no_delay, delay_preds)
    result = {"lookback_s": lookback,
              "noise_type": type(noiser).__name__,
              "noise_intensity_s": noiser.noise_intensity,
              "frequency_type": type(frequency).__name__,
              "frequency_s": frequency.frequency,
              "sensitive": sensitive}
    result.update(measure.point_metrics())
    return (result)


def experiment(name
               , lookback_range: List[int] = [600]
               , noise_intensity_range: List[int] = [600]
               , noise_types: List[Delayer] = [ExponentialDelayer]
               , frequency_range: List[DataFrequency] = [PoissonProcess(180)]
               , sensitivities: List[bool] = [True]):
    metrics = []
    print(f"Running experiment {name}")
    for lookback in lookback_range:
        for noise_intensity in noise_intensity_range:
            for noise_type in noise_types:
                for frequency in frequency_range:
                    for sensitivity in sensitivities:
                        metrics.append(run_one(lookback, frequency, noise_intensity, noise_type, sensitivity))
    df = pd.DataFrame.from_records(metrics)
    df.to_csv(name + ".csv", index=False)
    return df


def main():
    experiment_name = "mega_experiment_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.mkdir(experiment_name)
    os.chdir(experiment_name)  # Sorry about changing dirs

    lookback_range = [60, 120, 300, 600, 1200, 3000, 6000]
    lookback_df = experiment("lookback", lookback_range=lookback_range)

    sensitivities = [True, False]
    sensitivity_df = experiment("sensitivities", lookback_range=lookback_range, sensitivities=sensitivities)

    noise_types = [ExponentialDelayer]
    noise_intensities = [60, 120, 300, 600, 1200, 3000, 6000]
    noise_df = experiment("delay", noise_intensity_range=noise_intensities, noise_types=noise_types)

    frequency_range = [PoissonProcess(60),
                       PoissonProcess(120),
                       PoissonProcess(180),
                       PoissonProcess(600),
                       PoissonProcess(1200)]
    freq_df = experiment("frequencies", frequency_range=frequency_range)

    # Interaction
    freq_delay_df = experiment("freq_and_delay",
                               frequency_range=frequency_range,
                               noise_intensity_range=noise_intensities,
                               noise_types=noise_types)

    os.chdir("..")


if __name__ == '__main__':
    main()
