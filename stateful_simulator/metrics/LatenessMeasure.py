from stateful_simulator.metrics.Metrics import Metrics, PointMetric

from typing import List

class LatenessMeasure(Metrics):

    def point_metrics(self) -> List[PointMetric]:
        return [self.max_delay(),
                self.average_delay(),
                self.average_num_obs_difference(),
                self.max_num_obs_difference(),
                self.average_expected_number_obs()]

    def max_delay(self) -> PointMetric:

        return PointMetric("max_delay",
                           "The max time a point is delayed",
                           max([pred.timestamp.time_difference() for pred in self.delay]))

    def average_delay(self) -> PointMetric:
        return PointMetric("average_delay",
                           "The average time the delayed points are delay",
                           sum([pred.timestamp.time_difference() for pred in self.delay])/len(self.delay))

    def average_num_obs_difference(self) -> PointMetric:
        return PointMetric("average_num_obs_difference",
                           "The average number of observation differences",
                           sum([self.no_delay[ix].num_points - delay_pred.num_points
                                for ix,delay_pred in enumerate(self.delay)])/len(self.delay))

    def max_num_obs_difference(self) -> PointMetric:
        return PointMetric("max_num_obs_difference",
                           "The average number of observation differences",
                           max([self.no_delay[ix].num_points - delay_pred.num_points
                                for ix,delay_pred in enumerate(self.delay)]))

    def average_expected_number_obs(self):
        return PointMetric("average_expected_number_obs",
                           "The expected number of observations, on average",
                           sum(obs.num_points for obs in self.no_delay)/len(self.no_delay))