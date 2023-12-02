from political_party_classifier_models import *
from nltk.corpus import stopwords

__author__ = "Eli Tiao, David Ponce De Leon"
__copyright__ = "Copyright 2023, Westmont College"
__credits__ = ["Eli Tiao", "David Ponce De Leon"]
__email__ = "jtiao@westmont.edu, dponcedeleon@westmont.edu"

speeches = {file_id: inaugural.raw(file_id) for file_id in inaugural.fileids()}
# Training data
data = [
    (speeches['1853-Pierce.txt'], 'Democratic'),
    (speeches['1857-Buchanan.txt'], 'Democratic'),
    (speeches['1861-Lincoln.txt'], 'Republican'),
    (speeches['1865-Lincoln.txt'], 'Republican'),
    (speeches['1869-Grant.txt'], 'Republican'),
    (speeches['1873-Grant.txt'], 'Republican'),
    (speeches['1877-Hayes.txt'], 'Republican'),
    (speeches['1881-Garfield.txt'], 'Republican'),
    (speeches['1885-Cleveland.txt'], 'Democratic'),
    (speeches['1889-Harrison.txt'], 'Republican'),
    (speeches['1893-Cleveland.txt'], 'Democratic'),
    (speeches['1897-McKinley.txt'], 'Republican'),
    (speeches['1901-McKinley.txt'], 'Republican'),
    (speeches['1905-Roosevelt.txt'], 'Republican'),
    (speeches['1909-Taft.txt'], 'Republican'),
    (speeches['1913-Wilson.txt'], 'Democratic'),
    (speeches['1917-Wilson.txt'], 'Democratic'),
    (speeches['1921-Harding.txt'], 'Republican'),
    (speeches['1925-Coolidge.txt'], 'Republican'),
    (speeches['1929-Hoover.txt'], 'Republican'),
    (speeches['1933-Roosevelt.txt'], 'Democratic'),
    (speeches['1937-Roosevelt.txt'], 'Democratic'),
    (speeches['1941-Roosevelt.txt'], 'Democratic'),
    (speeches['1945-Roosevelt.txt'], 'Democratic'),
    (speeches['1949-Truman.txt'], 'Democratic'),
    (speeches['1953-Eisenhower.txt'], 'Republican'),
    (speeches['1957-Eisenhower.txt'], 'Republican'),
    (speeches['1961-Kennedy.txt'], 'Democratic'),
    (speeches['1965-Johnson.txt'], 'Democratic'),
    (speeches['1969-Nixon.txt'], 'Republican'),
    (speeches['1973-Nixon.txt'], 'Republican'),
    (speeches['1977-Carter.txt'], 'Democratic'),
    (speeches['1981-Reagan.txt'], 'Republican'),
    (speeches['1985-Reagan.txt'], 'Republican'),
    (speeches['1989-Bush.txt'], 'Republican'),
    (speeches['1993-Clinton.txt'], 'Democratic'),
    (speeches['1997-Clinton.txt'], 'Democratic'),
    (speeches['2001-Bush.txt'], 'Republican'),
    (speeches['2005-Bush.txt'], 'Republican'),
    (speeches['2009-Obama.txt'], 'Democratic'),
    (speeches['2013-Obama.txt'], 'Democratic'),
    (speeches['2017-Trump.txt'], 'Republican'),
    (speeches['2021-Biden.txt'], 'Democratic'),
]
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

test_data = [
    speeches['1877-Hayes.txt'],
    speeches['1881-Garfield.txt'],
    speeches['1885-Cleveland.txt'],
    speeches['1889-Harrison.txt'],
    speeches['1893-Cleveland.txt'],
    speeches['1897-McKinley.txt'],
    speeches['1901-McKinley.txt'],
    speeches['1905-Roosevelt.txt'],
    speeches['1909-Taft.txt'],
    speeches['1913-Wilson.txt'],
    speeches['1917-Wilson.txt'],
    speeches['1921-Harding.txt'],
    speeches['1925-Coolidge.txt'],
    speeches['1929-Hoover.txt'],
    speeches['1933-Roosevelt.txt'],
    speeches['1937-Roosevelt.txt'],
    speeches['1941-Roosevelt.txt'],
    speeches['1945-Roosevelt.txt'],
    speeches['1949-Truman.txt'],
    speeches['1953-Eisenhower.txt'],
    speeches['1957-Eisenhower.txt'],
    speeches['1961-Kennedy.txt'],
    speeches['1965-Johnson.txt'],
    speeches['1969-Nixon.txt'],
    speeches['1973-Nixon.txt'],
    speeches['1977-Carter.txt'],
    speeches['1981-Reagan.txt'],
    speeches['1985-Reagan.txt'],
    speeches['1989-Bush.txt'],
    speeches['1993-Clinton.txt'],
    speeches['1997-Clinton.txt'],
    speeches['2001-Bush.txt'],
    speeches['2005-Bush.txt'],
    speeches['2009-Obama.txt'],
    speeches['2013-Obama.txt'],
    speeches['2017-Trump.txt'],
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



top = classifier.present_features(50)
print(top)

for speech in test_data:
    test_feature_set = OurFeatureSet.build(speech, stop_words=stop_words)
    predicted_party = classifier.gamma(test_feature_set)
    print(f"The predicted party for the speech in {speech[:10]} is: {predicted_party}")
