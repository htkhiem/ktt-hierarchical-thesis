"""Implementation of the tfidf + leaf SGD classifier model."""
import os
import joblib
from importlib import import_module

import numpy as np
import pandas as pd

from sklearn import preprocessing, linear_model
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType

from models import model_sklearn
from utils.encoders.snowballstemmer import SnowballStemmerPreprocessor

REFERENCE_SET_FEATURE_POOL = 64


class Tfidf_LSGD(model_sklearn.SklearnModel):
    """A wrapper class around the scikit-learn-based tfidf-LSGD model.

    It exposes the same method signatures as the PyTorch-based models for
    ease of use in the main training controller.
    """

    def __init__(self, hierarchy=None, config=None, verbose=False):
        """Construct the TF-IDF + LeafSGD model.

        Parameters
        ----------
        hierarchy: None
            This model does not use the hierarchy parameter.
        config : dict
            A configuration dictionary. See the corresponding docs section for
            fields used by this model.
        verbose: None
            This model does not have additional printing options.
        """
        if hierarchy is not None:
            clf = linear_model.SGDClassifier(
                loss=config['loss'],
                max_iter=config['max_iter']
            )
            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(min_df=config['min_df'])),
                ('clf', clf),
            ])
        else:
            self.pipeline = None
        self.config = config

    @classmethod
    def from_checkpoint(cls, path):
        """Construct model from saved checkpoints as produced by previous\
        instances of this model.

        Parameters
        ----------
        path : str
            Path to the checkpoint. Checkpoints have a `.pt` extension.

        Returns
        -------
        instance : Tfidf_LSGD
            An instance that fully replicates the one producing the checkpoint.

        See also
        --------
        save : Create a checkpoint readable by this method.
        load : An alternative to this method for already-constructed instances.
        """
        instance = cls()
        instance.load(path)
        return instance

    @classmethod
    def get_preprocessor(cls, config):
        """Return a SnowballStemmere instance for this model."""
        return SnowballStemmerPreprocessor(config)

    def save(self, path, optim=None, dvc=True):
        """Serialise the pipeline into a pickle.

        The model state is saved as a `.pt` checkpoint with all the weights
        as well as its code (Scikit-learn only). See the related docs section
        for the schema of checkpoints produced by this model.

        This method is mostly used internally but could be of use in case
        one prefers to implement a custom training routine.

        Parameters
        ----------
        path : str
             Path to save the checkpoint file to.
        dvc : bool
            Whether to add saved checkpoints to Data Version Control.
        """
        joblib.dump(self.pipeline, path)

        if dvc:
            os.system('dvc add ' + path)

    def load(self, path):
        """Load saved pipeline pickle.

        Scikit-learn models also serialise their own code, so this is
        equivalent to the ``from_checkpoint`` classmethod in PyTorch models.

        Parameters
        ----------
        path : str
             Path to the checkpoint file.
        """
        self.pipeline = joblib.load(path)

    def fit(
            self,
            train_loader,
            val_loader=None,  # Unused but included for signature compatibility
            path=None,
            best_path=None,
            dvc=True
    ):
        """Train this TF-IDF + LeafSGD pipeline. No validation phase.

        Parameters
        ----------
        train_loader : tuple
            A data tuple generated by model_sklearn.get_loaders() to be used
            as the training set. The tuple contains a set of stemmed text as
            training input and a set of label.
        path : str, optional
            Path to save the latest epoch's checkpoint to. If this or
            `best_path` is unspecified, no checkpoint will be saved (dry-run).
        best_path: str, optional
            Path to separately save the best-performing epoch's checkpoint to.
            If this or `path` is unspecified, no checkpoint will be saved
            (dry-run).
        dvc : bool
            Whether to add saved checkpoints to Data Version Control.
        """
        self.pipeline.fit(train_loader[0], train_loader[1])
        if path is not None or best_path is not None:
            # There's no distinction between path and best_path as there is
            # no validation phase.
            self.save(path if path is not None else best_path, dvc)
        return None

    def test(self, loader, return_encodings=False):
        """Test this model on a dataset.

        This method can be used to run this instance (trained or not) over any
        dataset wrapped in a suitable test set tuple.

        Parameters
        ----------
        loader : tuple
            A data tuple generated by model_sklearn.get_loaders() to be used
            as the testing set. The tuple contains a set of stemmed text as
            testing input and a set of label.
        return_encodings: bool
            If true, return pooled tfidf encodings along. Useful for generating
            reference datasets.

        Returns
        -------
        test_output: dict
            The output of the testing phase, containing four metrics which can
            be retrived by model_sklearn.get_metrics(): targets, targets_b
            (binarized label), predictions and scores.
        """
        y_avg = preprocessing.label_binarize(
            loader[1],
            classes=self.pipeline.classes_
        )
        tfidf_encoding = self.pipeline.steps[0][1].transform(loader[0])
        scores = self.pipeline.steps[1][1].predict_proba(tfidf_encoding)
        predictions = [
            self.pipeline.classes_[i]
            for i in np.argmax(scores, axis=1)
        ]

        res = {
            'targets': loader[1],
            'targets_b': y_avg,
            'predictions': predictions,
            'scores': scores,
        }
        if return_encodings:
            pooled_feature_size = len(self.pipeline.steps[0][1].vocabulary_) \
                // REFERENCE_SET_FEATURE_POOL
            # Average-pool encodings
            tfidf_encoding_dense = tfidf_encoding.toarray()
            res['encodings'] = np.array([
                [
                    np.average(
                        tfidf_encoding_dense[
                            j,
                            i*REFERENCE_SET_FEATURE_POOL:
                            min((i+1)*REFERENCE_SET_FEATURE_POOL, len(scores))
                        ]
                    )
                    for i in range(0, pooled_feature_size)
                ]
                for j in range(tfidf_encoding_dense.shape[0])
            ])
        return res

    def gen_reference_set(self, loader):
        """Generate an Evidently-compatible reference dataset.

        Due to the data drift tests' computational demands, we only record
        average-pooled features.

        Parameters
        ----------
        loader: torch.utils.data.DataLoader
            A DataLoader wrapping around a dataset to become the reference
            dataset (ideally the test set).
        Returns
        -------
        reference_set: pandas.DataFrame
            An Evidently-compatible dataset. Numerical feature column names are
            simple stringified numbers, while targets are leaf classes' string
            names.
        """
        results = self.test(loader, return_encodings=True)
        pooled_features = results['encodings']
        scores = results['scores']
        targets = loader[1]
        scores = results['scores']
        cols = {
            'targets': targets,
        }
        for col_idx in range(pooled_features.shape[1]):
            cols['F' + str(col_idx)] = pooled_features[:, col_idx]
        for col_idx in range(scores.shape[1]):
            cols['C' + str(self.pipeline.classes_[col_idx])] =\
                scores[:, col_idx]
        return pd.DataFrame(cols)

    def export_onnx(self, classifier_path, encoder_path=None):
        """Export this model as an ONNX graph or two.

        Parameters
        ----------
        classifier_path: str
            Where to write the classifier head to.
        encoder_path: None
            This model does not export its encoder (tfidf) separately.
        """
        initial_type = [('str_input', StringTensorType([None, 1]))]
        onx = to_onnx(
            self.pipeline, initial_types=initial_type, target_opset=11
        )
        # Export
        with open(classifier_path + 'classifier.onnx', "wb") as f:
            f.write(onx.SerializeToString())

    def export_bento_resources(self, svc_config={}):
        """Export the necessary resources to build a BentoML service of this \
        model.

        Parameters
        ----------
        config: dict
            Additional configuration to pack into the BentoService.

        Returns
        -------
        config: dict
            Evidently configuration data specific to this instance.
        svc: BentoService subclass
            A fully packed BentoService.
        """
        # Config for monitoring service
        config = {
            'prediction': [
                'C' + str(i) for i in self.pipeline.classes_
            ]
        }
        svc_lts = import_module('models.tfidf_lsgd.bentoml.svc_lts')
        svc = svc_lts.Tfidf_LSGD()
        svc.pack('model', self.pipeline)
        svc.pack('config', svc_config)

        return config, svc
