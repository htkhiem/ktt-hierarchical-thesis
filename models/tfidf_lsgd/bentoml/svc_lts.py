"""Service file for Tfidf-LeafSGD."""
import os
import requests
from typing import List
import json

import numpy as np

import bentoml
from bentoml.adapters import JsonInput
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.service.artifacts.common import JSONArtifact
from bentoml.types import JsonSerializable

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
# These can't be put inside the class since they don't have _unload(), which
# prevents joblib from correctly parallelising the class if included.
STOP_WORDS = set(stopwords.words('english'))

EVIDENTLY_HOST = os.environ.get('EVIDENTLY_HOST', 'localhost')
EVIDENTLY_PORT = os.environ.get('EVIDENTLY_PORT', 5001)

REFERENCE_SET_FEATURE_POOL = 64

@bentoml.env(
    requirements_txt_file='models/db_bhcn/bentoml/requirements.txt',
    docker_base_image='bentoml/model-server:0.13.1-py36-gpu'
)
@bentoml.artifacts([
    SklearnModelArtifact('model'),
    JSONArtifact('config'),
])
class Tfidf_LSGD(bentoml.BentoService):
    """Real-time inference service for Tf-idf+LeafSGD."""

    _initialised = False

    def init_fields(self):
        """Initialise the necessary fields. This is not a constructor."""
        self.model = self.artifacts.model
        # Load service configuration JSON
        self.monitoring_enabled = self.artifacts.config['monitoring_enabled']
        self.pooled_feature_size = self.model.n_features_in_ // REFERENCE_SET_FEATURE_POOL

        self._initialised = True

    @bentoml.api(
        input=JsonInput(),
        batch=True,
        mb_max_batch_size=64,
        mb_max_latency=2000,
    )
    def predict(self, parsed_json_list: List[JsonSerializable]):
        """Classify text to the trained hierarchy."""
        if not self._initialised:
            self.init_fields()
        tokenized = [word_tokenize(j['text']) for j in parsed_json_list]
        stemmed = [
            ' '.join([stemmer.stem(word) if word not in STOP_WORDS else word for word in lst])
            for lst in tokenized
        ]
        tfidf_encoding = model.steps[0].transform(stemmed)
        scores = model.steps[1].steppredict_proba(tfidf_encoding)
        predictions = [model.classes_[i] for i in np.argmax(scores, axis=1)]

        if self.monitoring_enabled:
            """
            Create a 2D list contains the following content:
            [:, 0]: leaf target names (left as zeroes)
            [:, 1:n]: pooled features,
            [:, n:]: leaf classification scores,
            where n is the number of pooled features.
            The first axis is the microbatch axis.
            """
            new_rows = np.zeros(
                (len(texts), 1 + self.pooled_feature_size + len(self.pipeline.classes_)),
                dtype=np.float64
            )
            new_rows[
                :,
                1:self.pooled_feature_size+1
            ] = np.array([
                np.average(
                    tfidf_encoding[
                        :,
                        i*REFERENCE_SET_FEATURE_POOL:
                        min((i+1)*REFERENCE_SET_FEATURE_POOL, len(scores))
                    ]
                )
                for i in range(0, pooled_feature_size)
            ])
            new_rows[
                :,
                self.pooled_feature_size+1:
            ] = scores
            requests.post(
                "http://{}:{}/iterate".format(EVIDENTLY_HOST, EVIDENTLY_PORT),
                data=json.dumps({'data': new_rows.tolist()}),
                headers={"content-type": "application/json"},
            )
        return predictions
