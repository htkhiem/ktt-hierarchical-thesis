"""
Service file for TFIDF-LSGD + Walmart_30k.
"""
# BentoML
import bentoml
from bentoml.io import Text

# The model stuff
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize


nltk.download('punkt')
nltk.download('stopwords')
# These can't be put inside the class since they don't have _unload(), which
# prevents joblib from correctly parallelising the class if included.
stop_words = set(stopwords.words('english'))

classifier_runner = bentoml.sklearn.load_runner(
    '${classifier}'
)

stemmer = SnowballStemmer('english')

svc = bentoml.Service(
    'tfidf_lsgd',
    runners=[classifier_runner]
)


# Init Evidently monitoring service
with open("evidently.yaml", 'rb') as evidently_config_file:
    evidently_config = yaml.safe_load(evidently_config_file)

 SERVICE = MonitoringService(
     reference_data,
     options=options,
     column_mapping=ColumnMapping(**config['column_mapping'])
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
