"""Unit tests for functions in `spider.orb.orb_models`.
"""

import unittest
from classifier.political_party_classifier.political_party_classifier_models import *

__author__ = "David Ponce De Leon, Eli Tiao"
__credits__ = ["David Ponce De Leon, Eli Tiao"]
__email__ = "dponcedeleon@westmont.edu, jtiao@westmont.edu"


class Test(unittest.TestCase):
    def test_build(self):
        source_object = "Some test data"
        known_class = "Positive"
        feature_set = OurFeatureSet.build(source_object, known_class)
        self.assertIsInstance(feature_set, FeatureSet)
        self.assertEqual(feature_set.clas, known_class)

        self.assertIsNotNone(feature_set.feat)
        self.assertGreater(len(feature_set.feat), 0)
        self.assertIn("feature_name", [feature.name for feature in feature_set.feat])

    def test_gamma(self):
        test_feature_set = OurFeatureSet({Feature("feature1", True), Feature("feature2", False)})
        classifier = OurAbstractClassifier.train([OurFeatureSet({...}), OurFeatureSet({...})])
        predicted_class = classifier.gamma(test_feature_set)
        self.assertIsInstance(predicted_class, str)

        self.assertTrue(predicted_class in ["Class1", "Class2", "Class3"])
        self.assertNotEqual(predicted_class, "Unknown")

    def test_present_features(self):
        classifier = OurAbstractClassifier.train([OurFeatureSet({...}), OurFeatureSet({...})])
        with self.assertLogs() as log:
            classifier.present_features(top_n=3)

        # Add more assertions based on the expected behavior of your present_features function
        self.assertEqual(len(log.records), 1)
        self.assertIn("Feature Ranking:", log.output[0])
        self.assertRegex(log.output[0], r"Top \d features:")

    def test_train(self):
        training_data = [
            OurFeatureSet({...}),
            OurFeatureSet({...}),
        ]
        classifier = OurAbstractClassifier.train(training_data)
        self.assertIsInstance(classifier, AbstractClassifier)

        self.assertIsNotNone(classifier)
        self.assertTrue(hasattr(classifier, 'gamma'))
        self.assertTrue(hasattr(classifier, 'present_features'))


if __name__ == '__main__':
    unittest.main()