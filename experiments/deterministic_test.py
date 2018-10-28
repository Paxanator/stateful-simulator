from stateful_simulator.models.SklearnModel import SklearnModel
from stateful_simulator.models.StatefulModel import StatefulModel
from stateful_simulator.time_delays.ExponentialDelayer import ExponentialDelayer
from stateful_simulator.data_generators.util_data import deterministic_dataset
from stateful_simulator.featurizers.AggFeaturizer import AggFeaturizer
from stateful_simulator.metrics.DefaultMeasures import DefaultMeasures

from sklearn.linear_model import LinearRegression
from datetime import timedelta
from random import random

def main():

    aggregator = AggFeaturizer(lambda x: min)
    dataset = deterministic_dataset(decision_function=lambda y: sum([(ix * x + 4) for ix, x in enumerate(y)]) + 4,
                                    random_error=random,
                                    featurizer=aggregator,
                                    lookback=timedelta(minutes=10),
                                    num_features=5,
                                    train_percentage=.5,
                                    num_points=1000,
                                    frequency=lambda: timedelta(minutes=random()*3)
                                    )

    delayer = ExponentialDelayer(1 / (60))
    delayed_dataset = delayer.add_delay(dataset)
    model = SklearnModel(LinearRegression())

    stateful_model = StatefulModel(model, aggregator, timedelta(minutes=10))
    stateful_model.train(delayed_dataset)
    no_delay = stateful_model.predict(delayed_dataset, False)
    delay_preds = stateful_model.predict(delayed_dataset, True)
    measure = DefaultMeasures(no_delay, delay_preds)
    for measure in measure.point_metrics():
        print(measure)

if __name__ == '__main__':
    main()
