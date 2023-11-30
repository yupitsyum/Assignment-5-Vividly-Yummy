import nltk
from nltk.corpus import inaugural
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import random
from ..classifier_models import *

__author__ = "Eli Tiao, David Ponce De Leon"
__copyright__ = "Copyright 2023, Westmont College"
__credits__ = ["Eli Tiao", "David Ponce De Leon"]
__email__ = "jtiao@westmont.edu, dponcedeleon@westmont.edu"


# makes a dictionary
class OurFeature(Feature):
    def __init__(self, speech):
        super().__init__()
        self.speech = speech

    def split(self):
        return word_tokenize(self.speech)

    def makeWordDict(self):
        words = self.split()
        word_features = {}

        for word in words:
            if word in word_features:
                word_features[word] += 1
            else:
                word_features[word] = 1

        return word_features


class OurFeatureSet(FeatureSet):
    """A set of features that represent a single object. Optionally includes the known class of the object.
        Our feature set is going to consist of the individual words within a chunk of each inaugural speech.

        Attributes:
            _feat (set[Feature]): a set of features that define this object for the purposes of a classifier
            _clas (str | None): optional attribute set as the pre-defined classification of this object
        """
    def __init__(self, features: set[Feature], known_clas=None ):
        super().__init__(features, known_clas)

    @classmethod
    def build(cls, source_object: Any, known_clas=None, **kwargs) -> FeatureSet:
        """Method that builds and returns an instance of FeatureSet given a source object that requires preprocessing.

        :param source_object: object to build the feature set from
        :param known_clas: pre-defined classification of the source object
        :param kwargs: any additional data needed to preprocess the `source_object` into a feature set
        :return: an instance of `FeatureSet` built based on the `source_object` passed in
        """
        # TODO: build sets of sentences that have similar features
        if known_clas is None:
            raise ValueError("known_clas must be provided to build")
        feature_instance = OurFeature(source_object)
        features = feature_instance.makeWordDict()

        return OurFeatureSet(features, known_clas)


class OurAbstractClassifier(AbstractClassifier):
    """After classifying our train set by hand, the abstract classifier will allow us to see which words can most
        accurately identify which party the speech is from. """
    def __init__(self):
        super().__init__()
        self.feature_freq_dist = FreqDist()
        self.total_samples = 0

    def gamma(self, a_feature_set: FeatureSet) -> str:
        """Given a single feature set representing an object to be classified, returns the most probable class
        for the object based on the training this classifier received (via a call to `train` class method).

        :param a_feature_set: a single feature set representing an object to be classified
        :return: name of the class with the highest probability for the object
        """
        # TODO: return probability for the sentence and the political party
        return str(max(a_feature_set._feat, key=lambda word: a_feature_set._feat[word]))

    def present_features(self, top_n: int = 1) -> None:
        """Prints `top_n` feature(s) used by this classifier in the descending order of informativeness of the
        feature in determining a class for any object. Informativeness of a feature is a quantity that represents
        how "good" a feature is in determining the class for an object.

        :param top_n: how many of the top features to print; must be 1 or greater
        """
        # TODO: present the features that were most helpful in determining the political party of the sentence
        top_features = self.feature_freq_dist.most_common(top_n)
        print(f"Top {top_n} features:")
        for feature, count in top_features:
            print(f"{feature}: {count}")

    @classmethod
    def train(cls, training_set: Iterable[FeatureSet]) -> AbstractClassifier:
        """Method that builds a Classifier instance with its training (supervised learning) already completed. That is,
        the `AbstractClassifier` instance returned as the result of invoking this method must support `gamma` and
        present_features` method calls immediately without needing any other method invocations prior to them.

        :param training_set: An iterable collection of `FeatureSet` to use for training the classifier
        :return: an instance of `AbstractClassifier` with its training already completed
        """
        # TODO: Implement it such that it takes in a feature set of sentences of either political party to train
        classifier = OurAbstractClassifier()

        for feature_set in training_set:
            classifier.feature_freq_dist.update(feature_set._feat)
            classifier.total_samples += 1

        return classifier
