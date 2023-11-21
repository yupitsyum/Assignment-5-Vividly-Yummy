import nltk
from nltk.corpus import inaugural
import random
from classifier_models import *

__author__ = "Eli Tiao, David Ponce De Leon"
__copyright__ = "Copyright 2023, Westmont College"
__credits__ = ["Eli Tiao", "David Ponce De Leon"]
__email__ = "jtiao@westmont.edu, dponcedeleon@westmont.edu"


class OurFeatureSet(FeatureSet):

    def build(cls, source_object: Any, known_clas=None, **kwargs) -> FeatureSet:
        pass


class OurAbstractClassifier(AbstractClassifier):
    def gamma(self, a_feature_set: FeatureSet) -> str:
        pass

    def present_features(self, top_n: int = 1) -> None:
        pass

    def train(cls, training_set: Iterable[FeatureSet]) -> AbstractClassifier:
        pass
