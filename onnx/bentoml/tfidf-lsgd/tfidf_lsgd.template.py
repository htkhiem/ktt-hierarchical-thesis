"""
Service file for TFIDF-LSGD + Walmart_30k.
"""

import bentoml
from bentoml.io import Text
import numpy as np
import onnxruntime
import nltk
import sklearn
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn import preprocessing, metrics
from sklearn.pipeline import make_pipeline, Pipeline


# Use load_runner for optimised performance
# The transformer encoder section uses a directly-exported model, instead of
# through ONNX, for optimisation.

# PyTorch-based



classifier_runner = bentoml.sklearn.load_runner(
    '${classifier}'
)

stemmer = SnowballStemmer('english')

svc = bentoml.Service(
    'tfidf_lsgd',
    runners=[encoder_runner, classifier_runner]
)


@svc.api(input=Text(), output=Text())
def predict(input_text: str) -> np.ndarray:
    """Define the entire inference process for this model."""
    words = word_tokenize(input_text)
    result_list = map(
            lambda word: (
                stemmer.stem(word)
                if word not in stop_words
                else word
            ),
            words
        )
    stemmed_words = ' '.join(result_list)
    classifier_outputs = classifier_runner.run(stemmed_words)
    return '\n'.join(classifier_output)



    
