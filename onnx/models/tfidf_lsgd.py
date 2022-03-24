"""Implementation of the tfidf + leaf SGD classifier model."""
import os
import pandas as pd

from sklearn import preprocessing, linear_model
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
import bentoml

from models import model, model_sklearn


class Tfidf_LSGD(model.Model):
    """
    A wrapper class around the scikit-learn-based tfidf-LSGD model.

    It exposes the same method signatures as the PyTorch-based models for
    ease of use in the main training controller.
    """

    def __init__(self, config=None, verbose=False):
        """Construct the classifier."""

        clf = linear_model.SGDClassifier(
            loss='modified_huber',
            verbose=True,
            max_iter=1000
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
        """Train this tfidf-LSGD model. No validation phase."""
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
        """Export model to ONNX/Bento."""
        initial_type = [('str_input', StringTensorType([None, 1]))]
        onx = to_onnx(
            self.pipeline, initial_types=initial_type, target_opset=11
        )
        # Create path
        name = '{}_{}'.format(
            'tfidf_lsgd',
            dataset_name
        )
        path = 'output/{}/classifier/'.format(name)

        if not os.path.exists(path):
            os.makedirs(path)

        path += 'classifier.onnx'

        # Clear previous versions
        if os.path.exists(path):
            os.remove(path)

        # Export
        with open(path, "wb") as f:
            f.write(onx.SerializeToString())

        # Bento support
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
