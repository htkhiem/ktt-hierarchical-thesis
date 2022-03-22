"""Implementation of the tfidf + leaf SGD classifier model."""
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

from sklearn import preprocessing, linear_model
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from tempfile import mkdtemp

from models import model

cachedir = mkdtemp()
nltk.download('punkt')
nltk.download('stopwords')
# These can't be put inside the class since they don't have _unload(), which
# prevents joblib from correctly parallelising the class if included.
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))


class ColumnStemmer(BaseEstimator, TransformerMixin):
    """Serialisable pipeline stage wrapper for NLTK SnowballStemmer."""

    def __init__(self, verbose=False):
        """Construct wrapper.

        Actual stemmer object is in global scope as it cannot be serialised.
        """
        self.verbose = verbose

    def stem_and_concat(self, text):
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

    def fit(self, x, y=None):
        """Do nothing. This is not a trainable stage."""
        return self

    def transform(self, series):
        """Call stem_and_concat on series."""
        if self.verbose:
            print('Stemming column', series.name)
        return series.apply(self.stem_and_concat)


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
            ('stemmer', ColumnStemmer(verbose=verbose)),
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
        raise RuntimeError
