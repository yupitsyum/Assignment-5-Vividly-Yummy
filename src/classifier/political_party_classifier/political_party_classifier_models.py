import nltk
from nltk.corpus import inaugural
import random
from classifier.classifier_models import *

__author__ = "Eli Tiao, David Ponce De Leon"
__copyright__ = "Copyright 2023, Westmont College"
__credits__ = ["Eli Tiao", "David Ponce De Leon"]
__email__ = "jtiao@westmont.edu, dponcedeleon@westmont.edu"


class OurFeature(Feature):
    pass


class OurFeatureSet(FeatureSet):
    """A set of features that represent a single object. Optionally includes the known class of the object.

        Attributes:
            _feat (set[Feature]): a set of features that define this object for the purposes of a classifier
            _clas (str | None): optional attribute set as the pre-defined classification of this object
        """

    @classmethod
    def build(cls, source_object: Any, known_clas=None, **kwargs) -> FeatureSet:
        """Method that builds and returns an instance of FeatureSet given a source object that requires preprocessing.

        For instance, a subclass of `FeatureSet` may be designed to take in a text file object as the `source_object`
        build features based on the tokens that are present in the text file. In this subclass, the logic for
        tokenization and instantiation of `Feature` objects based on the tokens should be written in this method.

        The `return` statement in the actual implementation of this method should simply be a call to the
        constructor where `features` argument is the set of `Feature` instances created within the implementation of
        this method.

        :param source_object: object to build the feature set from
        :param known_clas: pre-defined classification of the source object
        :param kwargs: any additional data needed to preprocess the `source_object` into a feature set
        :return: an instance of `FeatureSet` built based on the `source_object` passed in
        """
        # TODO: build sets of sentences that have similar features
        pass


class OurAbstractClassifier(AbstractClassifier):
    """Abstract definition for an object classifier."""
    def __init__(self):
        self._struct1 = 0
        self._struct2 = 0

    @property
    def struct1(self):
        return self._struct1

    @property
    def struct2(self):
        return self._struct2

    def gamma(self, a_feature_set: FeatureSet) -> str:
        """Given a single feature set representing an object to be classified, returns the most probable class
        for the object based on the training this classifier received (via a call to `train` class method).

        :param a_feature_set: a single feature set representing an object to be classified
        :return: name of the class with the highest probability for the object
        """
        # TODO: return probability for the sentence and the political party
        pass

    def present_features(self, top_n: int = 1) -> None:
        """Prints `top_n` feature(s) used by this classifier in the descending order of informativeness of the
        feature in determining a class for any object. Informativeness of a feature is a quantity that represents
        how "good" a feature is in determining the class for an object.

        :param top_n: how many of the top features to print; must be 1 or greater
        """
        # TODO: present the features that were most helpful in determining the political party of the sentence
        pass

    @classmethod
    def train(cls, training_set: Iterable[FeatureSet]) -> AbstractClassifier:
        """Method that builds a Classifier instance with its training (supervised learning) already completed. That is,
        the `AbstractClassifier` instance returned as the result of invoking this method must support `gamma` and
        present_features` method calls immediately without needing any other method invocations prior to them.

        :param training_set: An iterable collection of `FeatureSet` to use for training the classifier
        :return: an instance of `AbstractClassifier` with its training already completed
        """
        # TODO: Implement it such that it takes in a feature set of sentences of either political party to train
        pass
