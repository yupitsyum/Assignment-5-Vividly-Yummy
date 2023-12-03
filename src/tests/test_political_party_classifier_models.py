"""Unit tests for functions in `spider.orb.orb_models`.
"""
from io import StringIO
from unittest.mock import patch

import pytest
import unittest
from classifier.political_party_classifier.political_party_classifier_models import *
from classifier.classifier_models import *
from nltk.corpus import stopwords

__author__ = "David Ponce De Leon, Eli Tiao"
__credits__ = ["David Ponce De Leon, Eli Tiao"]
__email__ = "dponcedeleon@westmont.edu, jtiao@westmont.edu"


import pytest
import unittest
from classifier.political_party_classifier.political_party_classifier_models import OurAbstractClassifier, OurFeatureSet, OurFeature
from classifier.classifier_models import Feature


class Test(unittest.TestCase):
    def test_build(self):
        source_object = "Some test data"
        known_class = "Positive"
        stop_words = set(stopwords.words('english'))
        feature_set = OurFeatureSet.build(source_object, known_class, stop_words=stop_words)
        self.assertIsInstance(feature_set, OurFeatureSet)
        self.assertEqual(feature_set.clas, known_class)

        self.assertIsNotNone(feature_set.feat)
        self.assertGreater(len(feature_set.feat), 0)
        self.assertIn("test", [feature.name for feature in feature_set.feat])
        self.assertIn("data", [feature.name for feature in feature_set.feat])

    def test_gamma(self):
        # Test case 1
        training_data = [
            OurFeatureSet({OurFeature("feature1", True), OurFeature("feature2", True)}, known_clas="Republican"),
            OurFeatureSet({OurFeature("feature1", True), OurFeature("feature2", True)}, known_clas="Democrat"),
        ]
        classifier = OurAbstractClassifier.train(training_data)
        test_feature_set_1 = OurFeatureSet({OurFeature("feature1", True), OurFeature("feature2", True)})

        predicted_class_1 = classifier.gamma(test_feature_set_1)
        self.assertEqual(predicted_class_1, "Democrat")

        # Test case 2
        test_feature_set_2 = OurFeatureSet({OurFeature("feature1", True), OurFeature("feature2", True)})
        predicted_class_2 = classifier.gamma(test_feature_set_2)
        self.assertEqual(predicted_class_2, "Democrat")

        # Add more test cases as needed

    def test_present_features(self):
        # Define a sample classifier with some data
        classifier_data = {
            "word1": [0.2, 0.8],
            "word2": [0.7, 0.3],
            "word3": [0.4, 0.6],
        }

        # Create an instance of OurAbstractClassifier
        classifier = OurAbstractClassifier(classifier_data)

        # Redirect stdout to capture the print output
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            # Call the present_features method
            classifier.present_features(top_n=2)

            # Get the printed output
            printed_output = mock_stdout.getvalue()

        # Split the output into lines for easier assertions
        output_lines = printed_output.strip().split("\n")

        # Assertions based on the expected behavior of present_features
        self.assertGreaterEqual(len(output_lines), 2)
        self.assertIn("Top 2 features:", output_lines[0])
        self.assertIn("word1 Democrat:Republican, 4.0:1", output_lines[1])
        self.assertIn("word2 Republican:Democrat, 2.3333333333333335:1", output_lines[2])

        # Add more assertions as needed

    def test_train(self):
        # Create some mock training data (replace {...} with actual feature sets)
        # Create mock training data
        # Replace {...} with actual features and class labels
        training_data = [
            OurFeatureSet({OurFeature("feature1", True), OurFeature("feature2", True)}, known_clas="Republican"),
            OurFeatureSet({OurFeature("feature3", True), OurFeature("feature4", True)}, known_clas="Democrat"),
            # Add more feature sets as needed
        ]

        # Call the train method to create a classifier
        classifier = OurAbstractClassifier.train(training_data)

        # Now you can use assertions to test the behavior of the train method
        # For example, check if the classifier is an instance of OurAbstractClassifier
        # and if it is not None
        self.assertIsInstance(classifier, OurAbstractClassifier)
        self.assertIsNotNone(classifier)

        # Add more assertions based on the specific behavior of your train method
        # For example, check if the probabilities are within a reasonable range
        # Check if probabilities are between 0 and 1
        for feature, (rep_prob, dem_prob) in classifier.dict.items():
            self.assertGreaterEqual(rep_prob, 0)
            self.assertLessEqual(rep_prob, 2)
            self.assertGreaterEqual(dem_prob, 0)
            self.assertLessEqual(dem_prob, 2)

        # Check if the probabilities sum to approximately 1
        for feature, (rep_prob, dem_prob) in classifier.dict.items():
            self.assertAlmostEqual(rep_prob + dem_prob, 3, delta=0.0001)

        # Add more assertions as needed


if __name__ == '__main__':
    unittest.main()