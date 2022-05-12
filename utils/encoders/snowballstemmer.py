import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

from .encoder import BasePreprocessor

SNOWBALLSTEMMER = None
STOPWORDS = None

class SnowballStemmerPreprocessor(BasePreprocessor):
    """A wrapper for SnowballStemmer with stopwords skipping."""

    def __init__(self, config):
        """Construct a preprocessor instance.

        All instances currently use the same SnowballStemmer/stopwords
        dictionary instance.
        """
        global SNOWBALLSTEMMER
        global STOPWORDS
        if SNOWBALLSTEMMER is None or STOPWORDS is None:
            nltk.download('punkt')
            nltk.download('stopwords')
            SNOWBALLSTEMMER = SnowballStemmer('english')
            STOPWORDS = set(stopwords.words('english'))

        super().__init__(config)

    def __call__(self, text):
        """Stem words that are not stopwords.

        Parameters
        ----------
        text: str
            The text to stem.

        Returns
        -------
        inputs: dict
            A dictionary containing the following field:

            - ``text``: Stemmed text
        """
        words = word_tokenize(text)
        result_list = map(
            lambda word: (
                SNOWBALLSTEMMER.stem(word)
                if word not in STOPWORDS
                else word
            ),
            words
        )
        return {'text': ' '.join(result_list)}


if __name__ == '__main__':
    pass
