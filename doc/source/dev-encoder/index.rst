.. Developing new encoders

Developing new encoders
=======================

KTT's encoders are just as modular as its classifiers. New encoders can be added to KTT as their own files, without affecting other code.

Where encoders come in
----------------------

KTT calls on encoder facilities in two different places:

1. In the dataset reading phase. Most encoders require special pre-processing to be done before the text themselves can be ingested by the actual encoder. KTT wraps the code that performs such tasks in ``Preprocessor`` classes, which are subclasses of the :py:class:`BasePreprocessor` class. Specifically, these are called for every requested row requested from the *intermediate datasets*, then either sent to the model directly or marshalled into minibatches by a framework-specific tool.

2. In the model itself. KTT models are combinations of encoders and classifiers. Each model instance contains its own encoder (or reference to a common, immutable encoder instance). Models also get to decide when and how to use their encoders by implementing their own ``forward`` method (or an equivalent of it, such as packaging their steps, encoder included, into a Scikit-learn ``Pipeline``).

Adding encoders
---------------

Trivial encoders such as Scikit-learn's ``feature_extraction`` tools can simply be called directly in the models that use them. What requires a bit of elbow grease are encoders that do not exist as libraries that you can call, or when you simply want your own implementation of them.

KTT currently does not provide a fixed encoder specification to allow for maximum flexibility in encoder and classifier design. This is also due to the fact that models are fixed encoder-classifier pairings that will not change at runtime. In other words, we do not support hot-swapping encoders as if they are a hyperparameter. Due to this rigidity, we can also assume that a classifier in a model knows how to call an encoder - the only one that it is implemented with. There are however some guidelines:

- Your encoder should be packaged into a single Python class, if possible.
- Its implementation should be located in ``utils/encoders/<encoder_name.py>``.
- If your encoder learns from the data (i.e. it has parameters and vocabularies, for example), it must be serialisable in one way or another. You are not forced to use Pickle (as we also leave checkpoint designs to your discretion), but it is recommended so as to stay in line with bundled models.
- You should implement your encoder in the same framework as that used by the classifier heads you intend to use it with. This is obviously not mandatory - you can use a Scikit-learn vectoriser with a PyTorch model with some clever programming in the ``fit``, ``save`` and ``load`` methods, for example.

You may also find it useful to implement helper functions that can fetch instances of your encoders from the Internet in your source file in place of a full reimplementation. This is the approach taken by KTT for transformer encoders from Huggingface (read: DistilBERT).

Implementing preprocessors
--------------------------

Preprocessors are not called by your models but by the KTT framework. As such, they must conform to the :py:class:`BasePreprocessor` specification above. Its implementation should also be located in ``utils/encoders``. You can either choose to implement it in the same source file as a custom encoder (if that encoder is the only one using this preprocessor) or in its own file (if it is a more general preprocessor that can be used by others). An example of the former is :py:class:`utils.encoders.distilbert.DistilBertPreprocessor`, which wraps a DistilBERT tokeniser and is only used by DistilBERT, while the latter includes :py:class:`utils.encoders.snowballstemmer.SnowballStemmerPreprocessor` which can be used by many text-based Scikit-learn feature extractors, such as ``TfidfVectorizer``.
