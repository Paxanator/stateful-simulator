from stateful_simulator.models.SklearnModel import SklearnModel
from stateful_simulator.models.StatefulModel import StatefulModel
from stateful_simulator.time_delays.ExponentialDelayer import ExponentialDelayer
from stateful_simulator.data_generators.util_data import get_data_set
from stateful_simulator.featurizers.AggFeaturizer import AggFeaturizer
from stateful_simulator.metrics.DefaultMeasures import DefaultMeasures

from sklearn.linear_model import LinearRegression
from datetime import timedelta

def main():
    dataset = get_data_set(
        dataset_url="https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv",
        timestamp_col="date",
        timestamp_format='%Y-%m-%d %H:%M:%S',
        target_col='Appliances',
        feature_cols=["lights"] + ["T" + str(room) for room in range(1,10)] + ["RH_" + str(room) for room in range(1,10)],
        train_percetange=0.5)
    delayer = ExponentialDelayer(60 * 60 * 24)
    delayed_dataset = delayer.add_delay(dataset)
    model = SklearnModel(LinearRegression())
    aggregator = AggFeaturizer(lambda x: max(x))
    stateful_model = StatefulModel(model, aggregator, timedelta(minutes=360))
    stateful_model.train(delayed_dataset)
    no_delay = stateful_model.predict(delayed_dataset, False)
    delay_preds = stateful_model.predict(delayed_dataset, True)
    measure = DefaultMeasures(no_delay, delay_preds)
    print(measure.point_metrics())

if __name__ == '__main__':
    main()
