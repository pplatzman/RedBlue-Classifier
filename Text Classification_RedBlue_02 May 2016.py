import os
import time

from bs4 import BeautifulSoup
from sklearn.datasets.base import Bunch
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import cross_validation as cv

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import lxml

## For data exploration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas.io.sql as pd_sql
import sqlite3 as sql

#CONNECTING TO THE DATASET
CORPUS_ROOT = "C:/cygwin64/home/Paul/virtualenvs/capstone/RedBlue/myclone/WritingFiles_Debates_April 28 16"
#You will have to insert your own path to the transcript data folder here.#

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

#print out the readme file
print dataset.DESCR
#Remember to create a README file and place it inside your CORPUS ROOT directory if you haven't already done so.

#print the number of records in the dataset
len(dataset.data)

#Checking out the data
print dataset.data[-5:]
print dataset.target[-5:]

#CONSTRUCT FEATURE EXTRACTION
#TfIdfVectorizer = CountVectorizer and TfIdfTransformer all in one step.
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(dataset.data)
print X_train_tfidf.shape

#Logistic Regression: Model fit, transform, and testing
splits     = cv.train_test_split(X_train_tfidf, dataset.target, test_size=0.2)
X_train, X_test, Y_train, Y_test = splits

model      = LogisticRegression()
model.fit(X_train, Y_train)

expected   = Y_test
predicted  = model.predict(X_test)

print classification_report(expected, predicted)
print metrics.confusion_matrix(expected, predicted)

#Logistic Regression: Predict on new data
docs_new = ['these are a bunch of randomly created examples', 'we would need to put our RSS data here', 'trump cruz sanders', 'Trump guns ISIS', 'welfare Sanders civil rights', 'sports alias useless']
X_new_tfidf = tfidf.transform(docs_new)

predicted = model.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, category))
