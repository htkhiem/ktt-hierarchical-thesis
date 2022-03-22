"""Implementation of the tfidf + hierarchical SGD classifier model."""
import os
import pandas as pd
import joblib
import logging

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

from sklearn import preprocessing, linear_model, metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import bentoml

from models import model
from utils.dataset import RANDOM_SEED, TRAIN_SET_RATIO, VAL_SET_RATIO

nltk.download('punkt')
nltk.download('stopwords')
# These can't be put inside the class since they don't have _unload(), which
# prevents joblib from correctly parallelising the class if included.
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))


def stem_series(series):
    """Call stem_and_concat on series."""
    def stem_and_concat(text):
        """Stem words that are not stopwords."""
        words = word_tokenize(text)
        result_list = map(
            lambda word: (
                stemmer.stem(word)
                if word not in stop_words
                else word
            ),
            words
        )
        return ' '.join(result_list)
    return series.apply(stem_and_concat)


def make_hierarchy(classes, depth=None, verbose=False):
    """Construct a special hierarchy data structure.

    This structure is for sklearn_hierarchical_classification only.
    """
    temp = {
        ROOT: set()
    }
    for i in range(0, len(classes)):
        path = classes[i]
        if len(path) != 0:
            if verbose:
                print(path)
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
    """
    Generate 'loaders' for scikit-learn models.

    Scikit-learn models simply read directly from lists. There is no
    special DataLoader object like for PyTorch.
    """
    data = pd.read_parquet(path)
    # Generate hierarchy
    hierarchy = make_hierarchy(data[class_col_name], depth, verbose)
    if not full_set:
        small_data = data.sample(frac=0.25, random_state=RANDOM_SEED)
        train_set = small_data.sample(
            frac=TRAIN_SET_RATIO,
            random_state=RANDOM_SEED
        )
        val_test_set = small_data.drop(train_set.index)
    else:
        train_set = data.sample(frac=TRAIN_SET_RATIO, random_state=RANDOM_SEED)
        val_test_set = data.drop(train_set.index)

    val_set = val_test_set.sample(
        frac=VAL_SET_RATIO / (1-TRAIN_SET_RATIO),
        random_state=RANDOM_SEED
    )
    test_set = val_test_set.drop(val_set.index)

    train_set = train_set.reset_index(drop=True)
    val_set = val_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)

    X_train = stem_series(train_set[input_col_name])
    X_test = stem_series(test_set[input_col_name])
    y_train = train_set[class_col_name].apply(
        lambda row: row[min(depth - 1, len(row) - 1)]
    )
    y_test = test_set[class_col_name].apply(
        lambda row: row[min(depth - 1, len(row) - 1)]
    )

    return (X_train, y_train), (X_test, y_test), hierarchy


class Tfidf_HSGD(model.Model):
    """
    A wrapper class around the scikit-learn-based tfidf-HSGD model.

    It exposes the same method signatures as the PyTorch-based models for
    ease of use in the main training controller.
    """

    def __init__(self, config=None, hierarchy=None, verbose=False):
        """Construct the classifier."""
        bclf = make_pipeline(linear_model.SGDClassifier(
            loss='modified_huber',
            class_weight='balanced',
        ))
        clf = HierarchicalClassifier(
            base_estimator=bclf,
            class_hierarchy=hierarchy,
        )
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(min_df=50)),
            ('clf', clf),
        ])
        self.config = config

    @classmethod
    def from_checkpoint(cls, path):
        """Construct model from saved checkpoint."""
        instance = cls()
        instance.pipeline = joblib.load(path)
        return instance

    def save(self, path, optim=None):
        """Serialise pipeline into a pickle."""
        joblib.dump(self.pipeline, path)

    def load(self, path):
        """Unpickle saved pipeline."""
        self.pipeline = joblib.load(path)

    def fit(
            self,
            train_loader,
            val_loader=None,  # Unused but included for signature compatibility
            path=None,
            best_path=None
    ):
        """Train this tfidf-HSGD model. No validation phase."""
        self.pipeline.fit(train_loader[0], train_loader[1])
        if path is not None or best_path is not None:
            # There's no distinction between path and best_path as there is
            # no validation phase.
            self.save(path if path is not None else best_path)
        return None

    def test(self, loader):
        """Test this model on a dataset."""
        y_avg = preprocessing.label_binarize(
            loader[1],
            classes=self.pipeline.classes_
        )
        predictions = self.pipeline.predict(loader[0])
        scores = self.pipeline.predict_proba(loader[0])

        return {
            'targets': loader[1],
            'targets_b': y_avg,
            'predictions': predictions,
            'scores': scores,
        }

    def export(self, dataset_name, bento=False):
        """
        Export this model to BentoML.

        Due to the usage of sklearn_hierarchical_classification, this model
        cannot be exported to ONNX format and only supports direct-to-BentoML
        exporting.
        For compatibility, the bento flag is still there but must always be set
        to True.
        Failure to do so will raise a RuntimeError.
        """
        if not bento:
            raise RuntimeError('Tfidf-HSGD does not support ONNX exporting!')

        # Create path
        name = '{}_{}'.format(
            'tfidf_hsgd',
            dataset_name
        )

        # Only support BentoML scikit-learn runner
        bentoml.sklearn.save(name, self.pipeline)


def get_metrics(test_output, display='log', compute_auprc=True):
    """Specialised metrics function for scikit-learn model."""
    leaf_accuracy = metrics.accuracy_score(
        test_output['targets'],
        test_output['predictions']
    )
    leaf_precision = metrics.precision_score(
        test_output['targets'],
        test_output['predictions'],
        average='weighted',
        zero_division=0
    )
    leaf_auprc = metrics.average_precision_score(
        test_output['targets_b'],
        test_output['scores'],
        average="micro"
    )
    if display == 'print' or display == 'both':
        print("Leaf accuracy: {}".format(leaf_accuracy))
        print("Leaf precision: {}".format(leaf_precision))
        print("Leaf AU(PRC): {}".format(leaf_auprc))
    if display == 'log' or display == 'both':
        logging.info("Leaf accuracy: {}".format(leaf_accuracy))
        logging.info("Leaf precision: {}".format(leaf_precision))
        logging.info("Leaf AU(PRC): {}".format(leaf_auprc))

    return (leaf_accuracy, leaf_precision, None, None, leaf_auprc)
