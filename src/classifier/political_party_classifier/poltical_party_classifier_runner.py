import nltk
from nltk.corpus import inaugural
import random

from political_party_classifier_models import *

__author__ = "Eli Tiao, David Ponce De Leon"
__copyright__ = "Copyright 2023, Westmont College"
__credits__ = ["Eli Tiao", "David Ponce De Leon"]
__email__ = "jtiao@westmont.edu, dponcedeleon@westmont.edu"

# Function to classify a speech based on the provided classifier
def classify_speech(classifier, speech):
    words = word_tokenize(speech)
    word_dict = FreqDist(words)
    feature_set = OurFeatureSet(word_dict, known_clas=None)
    return classifier.gamma(feature_set)

# Load the speeches from the inaugural corpus
speeches = {}
for file_id in inaugural.fileids():
    words = inaugural.words(file_id)
    speech = ' '.join(words)
    speeches[file_id] = speech
print("speech ok")

# Training set - manually classify some speeches as Republican or Democratic

training_data = [
    (speeches['1929-Hoover.txt'], 'Republican'),
    (speeches['1953-Eisenhower.txt'], 'Republican'),
    (speeches['1921-Harding.txt'], 'Republican'),
    (speeches['1933-Roosevelt.txt'], 'Democratic'),
    (speeches['1949-Truman.txt'], 'Democratic'),
    (speeches['1961-Kennedy.txt'], 'Democratic'),

]

print("training data ok")

# Convert training data into feature sets
training_feature_sets = [OurFeatureSet.build(speech, known_clas=party) for speech, party in training_data]

# Train the classifier
classifier = OurAbstractClassifier.train(training_feature_sets)

# Test set - classify speeches using the trained classifier
test_data = [
    speeches['1801-Jefferson.txt'],
    speeches['1865-Lincoln.txt'],

]



for speech_id, speech_text in speeches.items():
    party = classify_speech(classifier, speech_text)
    print(f"Speech '{speech_id}' is classified as: {party}")
