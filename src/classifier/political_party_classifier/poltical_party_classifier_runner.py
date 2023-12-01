from political_party_classifier_models import *
from nltk.corpus import stopwords

__author__ = "Eli Tiao, David Ponce De Leon"
__copyright__ = "Copyright 2023, Westmont College"
__credits__ = ["Eli Tiao", "David Ponce De Leon"]
__email__ = "jtiao@westmont.edu, dponcedeleon@westmont.edu"

speeches = {file_id: inaugural.raw(file_id) for file_id in inaugural.fileids()}
# Training data
training_data = [
    (speeches['1933-Roosevelt.txt'], 'Democrat'),
    (speeches['1929-Hoover.txt'], 'Republican'),
    (speeches['1953-Eisenhower.txt'], 'Republican'),
    (speeches['1921-Harding.txt'], 'Republican'),
    (speeches['1949-Truman.txt'], 'Democrat'),
    (speeches['1961-Kennedy.txt'], 'Democrat'),
    (speeches['1841-Harrison.txt'], 'Republican'),
    (speeches['2001-Bush.txt'], 'Republican'),
    (speeches['2021-Biden.txt'], 'Democrat')
]

# Define stop words
stop_words = set(stopwords.words('english'))

# Create a list of training feature sets
training_feature_sets = [OurFeatureSet.build(speech, known_clas=party, stop_words=stop_words) for speech, party in
                         training_data]
i = 0
for index in training_feature_sets:
    print(training_feature_sets[i].clas)
    i += 1

# Train the classifier
classifier = OurAbstractClassifier.train(training_feature_sets)

# Test the classifier on a new speech
test_data = speeches['1801-Jefferson.txt']
test_feature_set = OurFeatureSet.build(test_data, stop_words=stop_words)
print(test_feature_set.feat)
print(test_feature_set.clas)

print(classifier.gamma(test_feature_set))

top = classifier.present_features(50)
print(top)

for file_id, speech in speeches.items():
    test_feature_set = OurFeatureSet.build(speech, stop_words=stop_words)
    predicted_party = classifier.gamma(test_feature_set)
    print(f"The predicted party for the speech in {file_id} is: {predicted_party}")
