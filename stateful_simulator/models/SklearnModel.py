
from stateful_simulator.models.StatelessModel import StatelessModel
from stateful_simulator.datatypes.DataTypes import FeatureVector
from typing import List, Tuple

from numpy import array, expand_dims

class SklearnModel(StatelessModel):

    def __init__(self, model):
        self.model = model

    def train(self, fvs: List[FeatureVector]):
        X, y = self._transform_to_sklearn_input(fvs)
        self.model.fit(X,y)

    def predict(self, fv: FeatureVector) -> float:
        return self.model.predict(expand_dims(array(fv.features),0))

    def _transform_to_sklearn_input(self, fvs:List[FeatureVector])-> Tuple[array,array]:
        return array([array(feature.features) for feature in fvs]), array([feature.target for feature in fvs])