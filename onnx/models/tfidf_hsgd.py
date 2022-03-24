"""Implementation of the tfidf + hierarchical SGD classifier model."""
import joblib

from sklearn import preprocessing, linear_model
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

import bentoml

from models import model


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
