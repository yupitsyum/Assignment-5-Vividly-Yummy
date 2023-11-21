"""Unit tests for functions in `spider.orb.orb_models`.
"""

import os
import unittest
from classifier.political_party_classifier_models import *

__author__ = "David Ponce De Leon, Eli Tiao"
__credits__ = ["David Ponce De Leon, Eli Tiao"]
__email__ = "dponcedeleon@westmont.edu, jtiao@westmont.edu"


class Test(unittest.TestCase):
    def test_build(self):
        source_object = "Some test data"
        known_class = "Positive"
        feature_set = FeatureSet.build(source_object, known_class)
        self.assertIsInstance(feature_set, FeatureSet)
        self.assertEqual(feature_set.clas, known_class)

        self.assertIsNotNone(feature_set.feat)
        self.assertGreater(len(feature_set.feat), 0)
        self.assertIn("feature_name", [feature.name for feature in feature_set.feat])

    def test_gamma(self):
        test_feature_set = FeatureSet({Feature("feature1", True), Feature("feature2", False)})
        classifier = OurAbstractClassifier.train([FeatureSet({...}), FeatureSet({...})])
        predicted_class = classifier.gamma(test_feature_set)
        self.assertIsInstance(predicted_class, str)

        # Add more assertions based on the expected behavior of your gamma function
        self.assertTrue(predicted_class in ["Class1", "Class2", "Class3"])
        self.assertNotEqual(predicted_class, "Unknown")
        self.assertRegex(predicted_class, r"^[A-Za-z0-9_]+$")

    def test_present_features(self):
        classifier = OurAbstractClassifier.train([FeatureSet({...}), FeatureSet({...})])
        with self.assertLogs() as log:
            classifier.present_features(top_n=3)

        # Add more assertions based on the expected behavior of your present_features function
        self.assertEqual(len(log.records), 1)
        self.assertIn("Feature Ranking:", log.output[0])
        self.assertRegex(log.output[0], r"Top \d features:")

    def test_train(self):
        training_data = [
            FeatureSet({...}),
            FeatureSet({...}),
            # Add more FeatureSets as needed for testing
        ]
        classifier = OurAbstractClassifier.train(training_data)
        self.assertIsInstance(classifier, AbstractClassifier)

        # Add more assertions based on the expected behavior of your train function
        self.assertIsNotNone(classifier)
        self.assertTrue(hasattr(classifier, 'gamma'))
        self.assertTrue(hasattr(classifier, 'present_features'))


if __name__ == '__main__':
    unittest.main()