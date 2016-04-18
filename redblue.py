import os
import time

from bs4 import BeautifulSoup
from sklearn.datasets.base import Bunch
from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import lxml

## CONNECTING TO THE DATASET

CORPUS_ROOT = "/Users/Goodgame/Desktop/Georgetown/RedBlue/rbdata/"

def load_data(root=CORPUS_ROOT):
    """
    Loads the text data into memory using the bundle dataset structure.
    Note that on larger corpora, memory safe CorpusReaders should be used.
    """

    # Open the README and store
    with open(os.path.join(root, 'README'), 'r') as readme:
        DESCR = readme.read()

    # Iterate through all the categories
    # Read the HTML into the data and store the category in target
    data      = []
    target    = []
    filenames = []

    for category in os.listdir(root):
        if category == "README": continue # Skip the README
        if category == ".DS_Store": continue # Skip the .DS_Store file
        for doc in os.listdir(os.path.join(root, category)):
            fname = os.path.join(root, category, doc)

            # Store information about document
            filenames.append(fname)
            target.append(category)

            # Read data and store in data list
            with open(fname, 'r') as f:
                data.append(f.read())

    return Bunch(
        data=data,
        target=target,
        filenames=filenames,
        target_names=frozenset(target),
        DESCR=DESCR,
    )

dataset = load_data()

# print out the readme file
print dataset.DESCR

## FEATURE extraction

class HTMLPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocesses HTML to extract the title and the paragraph.
    """

    def fit(self, X, y=None):
        return self

    def parse_body_(self, soup):
        """
        Helper function for dealing with the HTML body
        """

        if soup.find('p'):
            # Use paragraph extraction
            return "\n\n".join([
                    p.text.strip()
                    for p in soup.find_all('p')
                    if p.text.strip()
                ])

        else:
            # Use raw text extraction
            return soup.find('body').text.strip()

    def parse_html_(self, text):
        """
        Helper function for dealing with an HTML document
        """
        soup  = BeautifulSoup(text, 'lxml')
        title = soup.find('title').text
        body  = self.parse_body_(soup)

        # Get rid of the soup
        soup.decompose()
        del soup

        return {
            'title': title,
            'body': body
        }

    def transform(self, texts):
        """
        Extracts the text from all the paragraph tags
        """
        return [
            self.parse_html_(text)
            for text in texts
        ]


class ValueByKey(BaseEstimator, TransformerMixin):
    """
    Extracts a value from a dictionary by key.
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, dicts):
        """
        Returns a list of values by key.
        """
        return [
            d[self.key] for d in dicts
        ]

## CONSTRUCT FEATURE EXTRACTION Pipeline

features = Pipeline([

    # Preprocess html to extract the text.
    ('preprocess', HTMLPreprocessor()),

    # Use FeatureUnion to combine title and body features
    ('html_union', FeatureUnion(

        # Create union of Title and Body
        transformer_list=[

            # Pipeline for Title  Extraction
            ('title', Pipeline([
                ('title_extract', ValueByKey('title')),
                ('title_tf', CountVectorizer(
                    max_features=4000, stop_words='english'
                )),
                # TODO: Add advanced TF parameters for better features
            ])),

            # Pipeline for Task Extraction
            ('body', Pipeline([
                ('body_extract', ValueByKey('body')),
                ('body_tfidf', TfidfVectorizer(stop_words='english')),
                # TODO: Add advanced TF-IDF parameters for better features
            ])),

        ],

        # weight components in FeatureUnion
        transformer_weights={
            'title': 0.45,
            'body':  0.55,
        },

    ))

])

start = time.time()
data  = features.fit_transform(dataset.data)

print (
    "Processed {} documents with {} features in {:0.3f} seconds"
    .format(data.shape[0], data.shape[1], time.time()-start)
)

feature_names = features.steps[1][1].transformer_list[0][1].steps[1][1].get_feature_names()
feature_names.extend(
    features.steps[1][1].transformer_list[1][1].steps[1][1].get_feature_names()
)

## TOPIC MODELING AND CLUSTERING

N_TOPICS    = 10
N_TOP_WORDS = 20

def model_topics(model, data, **kwargs):
    """
    Automatic topic modeling and elucidation of topic classes.
    """

    start = time.time()
    clust = model(**kwargs).fit(data)

    print "Fit {} model in {:0.3f} seconds\n".format(clust.__class__.__name__, time.time()-start)
    for idx, topic in enumerate(clust.components_):
        print "  Topic {}:".format(idx)
        for tdx in topic.argsort()[:-N_TOP_WORDS - 1:-1]:
            print "    - {}".format(feature_names[tdx])
        print

model_topics(NMF, data, n_components=N_TOPICS, random_state=1, alpha=.1, l1_ratio=.5)

model_topics(LatentDirichletAllocation, data, n_topics=N_TOPICS, max_iter=5,
                                learning_method='online', learning_offset=50.,
                                random_state=0)

## TOPIC IDENTIFICATION

def classify_topics(model, data, **kwargs):
    start = time.time()
    clf = model(**kwargs).fit(data, dataset.target)

    print "Fit {} model in {:0.3f} seconds\n".format(clf.__class__.__name__, time.time()-start)
    classification_report(clf)
