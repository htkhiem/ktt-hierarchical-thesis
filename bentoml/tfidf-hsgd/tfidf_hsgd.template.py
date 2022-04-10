"""
Service file for TFIDF-HSGD + Walmart_30k.
"""

import bentoml
from bentoml.io import Text
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

classifier_runner = bentoml.sklearn.load_runner(
    '${classifier}'
)

stemmer = SnowballStemmer('english')

svc = bentoml.Service(
    'tfidf_hsgd',
    runners=[classifier_runner]
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
    return classifier_outputs.tostring()