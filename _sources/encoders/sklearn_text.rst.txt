Scikit-learn text feature extractors
====================================

KTT can provide facilities for using most Scikit-learn text feature extractors with your classifiers. Two of the bundled models also make use of Scikit-learn's ``TfidfVectorizer`` facilities.

At present, KTT comes bundled with one predesigned Scikit-learn preprocessor, the ``SnowballStemmerPreprocessor``. As its name implies, this preprocessor stems non-stopword tokens in your text to reduce the amount of vocabulary your encoder has to learn. Its stop words dictionary is powered by NLTK.

API
---

.. automodule:: utils.encoders.snowballstemmer
	:members:
	:special-members:

