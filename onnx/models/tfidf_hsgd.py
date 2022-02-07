import pandas as pd
from random import shuffle
import joblib
import logging

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn import preprocessing

from sklearn import linear_model, preprocessing, metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from tempfile import mkdtemp

from utils.dataset import RANDOM_SEED, TRAIN_SET_RATIO, VAL_SET_RATIO

cachedir = mkdtemp()
nltk.download('punkt')
nltk.download('stopwords')
# These can't be put inside the class since they don't have _unload(), which prevents
# joblib from correctly parallelising the class if included.
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

class ColumnStemmer(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def stem_and_concat(self, text):
        words = word_tokenize(text)
        result_list = map(lambda word: stemmer.stem(word) if word not in stop_words else word, words)
        return ' '.join(result_list)

    def fit(self, x, y=None):
        return self

    def transform(self, series):
        if self.verbose:
            print('Stemming column', series.name)
        return series.apply(self.stem_and_concat)

# Special hierarchy data structure for sklearn_hierarchical_classification
def make_hierarchy(classes, depth=None, verbose=False):
    temp = { ROOT: set()}
    for i in range(0, len(classes)):
        path = classes[i]
        if len(path) != 0:
            if verbose: print(path)
            limit = min(depth, len(path))
            if path[0] not in temp[ROOT]:
                temp[ROOT].add(path[0])
            for i in range(0, limit - 1):
                if path[i] not in temp:
                    temp[path[i]] = set()
            # add leaf into one of the generated sub-dicts.
            try:
                temp[path[limit-2]].add(path[limit-1])
            except:
                pass

        hierarchy = {}
        for key in temp.keys():
            hierarchy[key] = list(temp[key])
        return hierarchy

# This model is sklearn-based, and so is its data handling procedures.
def get_loaders(
        path,
        config,
        depth=2,
        full_set=True,
        input_col_name='title',
        class_col_name='category',
        partial_set_frac=0.05,
        verbose=False
):
    data = pd.read_parquet(path)
    # Generate hierarchy
    hierarchy = make_hierarchy(data[class_col_name], depth, verbose)
    if not full_set:
        small_data = data.sample(frac = 0.25, random_state=RANDOM_SEED)
        train_set = small_data.sample(frac = TRAIN_SET_RATIO, random_state=RANDOM_SEED)
        val_test_set = small_data.drop(train_set.index)
    else:
        train_set = data.sample(frac = TRAIN_SET_RATIO, random_state=RANDOM_SEED)
        val_test_set = data.drop(train_set.index)


    val_set = val_test_set.sample(frac = VAL_SET_RATIO / (1-TRAIN_SET_RATIO), random_state=RANDOM_SEED)
    test_set = val_test_set.drop(val_set.index)

    train_set = train_set.reset_index(drop=True)
    val_set = val_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)

    X_train = train_set[input_col_name]
    X_test = test_set[input_col_name]
    y_train = train_set[class_col_name].apply(lambda row: row[min(depth - 1, len(row) - 1)])
    y_test = test_set[class_col_name].apply(lambda row: row[min(depth - 1, len(row) - 1)])

    return (X_train, y_train), (X_test, y_test), hierarchy

# This thing doesn't use config but to maintain the same signature it's receiving one anyway.
def gen(config, hierarchy, verbose=False):
    bclf = make_pipeline(linear_model.SGDClassifier(
        loss='modified_huber', # Good perf in papers
        class_weight='balanced', # To fix our gross category example count imbalance
    ))
    clf = HierarchicalClassifier(
        base_estimator=bclf,
        class_hierarchy=hierarchy,
    )
    return Pipeline([
        ('stemmer', ColumnStemmer(verbose=verbose)),
        ('tfidf', TfidfVectorizer(min_df=50)),
        # Use a SVC classifier on the combined features
        ('clf', clf),
    ])

# This thing doesn't use val_loader but to maintain the same signature it's receiving one anyway.
def train(config, train_loader, val_loader, gen_model, path=None, best_path=None):
    pipeline = gen_model()
    pipeline.fit(train_loader[0], train_loader[1])
    if path is not None or best_path is not None:
        joblib.dump(pipeline, path if path is not None else best_path)
    return pipeline, None

def test(pipeline, config, loader):
    y_avg = preprocessing.label_binarize(loader[1], classes = pipeline.classes_)
    predictions = pipeline.predict(loader[0])
    scores = pipeline.predict_proba(loader[0])

    return {
        'targets': loader[1],
        'targets_b': y_avg,
        'predictions': predictions,
        'scores': scores,
    }

def get_metrics(test_output, display='log', compute_auprc=True):
    leaf_accuracy = metrics.accuracy_score(test_output['targets'], test_output['predictions'])
    leaf_precision = metrics.precision_score(test_output['targets'], test_output['predictions'], average='weighted', zero_division=0)
    leaf_auprc = metrics.average_precision_score(test_output['targets_b'], test_output['scores'], average = "micro")
    if display == 'print' or display == 'both':
        print("Leaf accuracy: {}".format(leaf_accuracy))
        print("Leaf precision: {}".format(leaf_precision))
        print("Leaf AU(PRC): {}".format(leaf_auprc))
    if display == 'log' or display == 'both':
        logging.info("Leaf accuracy: {}".format(leaf_accuracy))
        logging.info("Leaf precision: {}".format(leaf_precision))
        logging.info("Leaf AU(PRC): {}".format(leaf_auprc))

    return (leaf_accuracy, leaf_precision, None, None, leaf_auprc)
