.. Dev section - Scikit-learn.

Implementing a model with Scikit-learn
===================================================

Scikit-learn (sklearn) is another toolkit that this system supports. It is mainly used for lightweight ML models that do not require GPUs.

This article will guide you through the process of implementing a model from scratch using KTT's facilities and sklearn. It will mainly focus on standout aspects (those that are unique to sklearn).

The model
---------

To keep everything simple and focus on the system integration aspects only, we will recreate tfidf+LinearSGD, which is the simpler of the two bundled sklearn models. It consists of an NLTK-powered word tokeniser+stemmer, a term frequency-inverse document frequency (tfidf) vectoriser, and a single custom classifier performing stochastic gradient descent usinga modified Huber loss. More information about this model can be found in :ref:`tfidf-lsgd-theory`.

Scikit-learn model module structure
-----------------------------------

Each sklearn model module ('module' for short) in KTT is a self-contained collection of implemented source code, metadata and configuration files. A module defines its own training, checkpointing and exporting procedures. It might also optionally implement a  BentoML service and configuration files for live inference using the integrated BentoML-powered inference system and monitoring using Prometheus/Grafana. The general folder tree of an sklearn model is as detailed in :ref:`model-struct`.

The source implementation itself must subclass the abstract :py:class:`models.model_sklearn.SklearnModel` class, which subclasses the abstract :py:class:`models.model.Model` class and pre-implements two of the abstract methods for you (:py:meth:`models.model.Model.get_dataloader_func` and :py:meth:`models.model.Model.metrics_func`).

Scikit-learn utilities
----------------------

KTT provides framework-specific utilities for common tasks such as loading data in and computing performance metrics. For Scikit-learn, see :ref:`sklearn-utils`.

The process
-----------

Folder structure
~~~~~~~~~~~~~~~~

Let's name our model ``testmodel`` for brevity. First, create these files folders in accordance with KTT's folder structure:

.. code-block:: bash

    models
    └── testmodel
        ├── __init__.py
        ├── bentoml
        │   ├── __init__.py
        │   ├── evidently.yaml
        │   ├── requirements.txt
        │   ├── dashboard.json
        │   └── svc_lts.py
        └── testmodel.py

You can simply create blank files for now. We will go into detail of each file soon.

Hyperparameters
~~~~~~~~~~~~~~~

Let's first determine which tunable hyperparameter our model has:

-  ``loss``: Which loss function to use for the SGD classifier. The possible options are ``hinge``, ``log``, ``modified_huber``, ``squared_hinge``, and ``perceptron`` (only classification losses are listed here - regression losses shouldn't be used) Here we shall default to ``modified_huber``.
- ``max_iter``: Upper limit of how many descent iterations will be performed. Setting this to low may prevent the model from converging. Here we default to 1000.
- ``min_df``: The minimum number of occurences a word must have in the dataset for it to be included in the tfidf vectoriser's vocabulary. We will default it to 50.

At least our model-specific hyperparameters will have to be present in the ``config`` dict that we will soon see in the upcoming parts.

Implementing the model
~~~~~~~~~~~~~~~~~~~~~~

From here on we will refer to files using their paths in relative to the ``testmodel`` folder.

In ``testmodel.py``, import the necessary libraries and define a concrete subclass of the ``SklearnModel`` abstract class:

.. code-block:: python

	import os
	import joblib
	import yaml

	import numpy as np
	import pandas as pd

	from sklearn import preprocessing, linear_model
	from sklearn.pipeline import Pipeline

	from sklearn.feature_extraction.text import TfidfVectorizer
	from skl2onnx import to_onnx
	from skl2onnx.common.data_types import StringTensorType

	from models import model_sklearn
	from utils.encoders.snowballstemmer import SnowballStemmerPreprocessor
	from utils.build import init_folder_structure
	from .bentoml import svc_lts


    class TestModel(model_sklearn.SklearnModel):
        """A wrapper class around the scikit-learn-based test model.

        It's basically a replica of the Tfidf-LeafSGD model bundled with KTT.
        """

        def __init__(self, config=None, verbose=False):
            pass

        @classmethod
        def from_checkpoint(cls, path):
            pass
            
        @classmethod
        def get_preprocessor(cls, config):
        		pass

        def save(self, path, optim=None, dvc=True):
            pass

        def load(self, path):
            pass

        def fit(
                self,
                train_loader,
                val_loader=None,  # Unused but included for signature compatibility
                path=None,
                best_path=None,
                dvc=True
        ):
            pass

        def test(self, loader):
            pass

        def generate_reference_set(self, loader):
            pass

		def export_onnx(self, classifier_path, encoder_path):
			pass
			
		def export_bento_resources(self, svc_config={}):
			pass

    if __name__ == "__main__":
        pass

You might notice that there are more methods than what is there in the ``Model`` abstract class. They are for reference dataset generation. Since we do not force every model to be able to export to our BentoML-based inference system with full monitoring capabilities, these methods are not defined in the abstract class. However, they will be covered in this guide for the sake of completeness.

Now we will go through the process of implementing each method.

.. note::

    We highly recommend writing documentation for your model as you implement each method.

    KTT's documentation system uses Sphinx but follows PEP 8's documentation strings standard, with Sphinx features exposed to the syntax via the ``numpydoc`` extension. In short, you can refer to `this style guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

    The below code listings will not include full documentation (only short summary docstrings) for brevity.

``__init__``
^^^^^^^^^^^^

Constructing an sklearn model in KTT is quite simple compared to PyTorch. One is recommended to package all components into *pipelines* for easier exporting and importing. Here we have two components: the tfidf vectoriser and the SGD classifier. The stemmer and tokeniser is not present since they have already been taken care of by KTT's default sklearn facilities at the data-loading level.

One point of difference in terms of design from PyTorch model is that Scikit-learn models can entirely serialise themselves without needing external configuration and hierarchical metadata to be stored along. To take advantage of this, we will package everything into a single ``Pipeline`` and later use ``joblib`` to pickle it. However, there's a catch: since we do not store those information separately, we cannot reuse them to instantiate this model through the normal constructor as with PyTorch. As a workaround, we set up the constructor such that it can tolerate having no arguments (and later call ``load`` on it). In this case, the constructor should create an empty model with no pipeline or config saved.

.. code-block:: python
    :dedent: 0

        def __init__(self, hierarchy=None, config=None, verbose=False):
        		# It is possible that the constructor will be called without
        		# any of the arguments (by the from_checkpoint constructor).
        		# In that case simply instantiate an empty class.
        		if hierarchy is not None and config is not None:
		        # The SGD classifier
		        clf = linear_model.SGDClassifier(
		            loss=config['loss'],
		            max_iter=config['max_iter']
		        )
		        # Package into pipeline
		        self.pipeline = Pipeline([
		            ('tfidf', TfidfVectorizer(config['min_df'])),
		            ('clf', clf),
		        ])
		        # Back up config for later use
		        self.config = config

``save``
^^^^^^^^

Due to how high-level sklearn can be, saving and loading models are a breeze compared to PyTorch. Sklearn models can be saved in whole (including their code) in a single file. As such, to save this model, we only need to use ``joblib`` to serialise the entire pipeline.

.. code-block:: python
    :dedent: 0

        def save(self, path, optim=None, dvc=True):
            joblib.dump(self.pipeline, path)

            if dvc:
                os.system('dvc add ' + path)

``load``
^^^^^^^^

The reverse is performed in this method compared to ``save``.

Thanks to how sklearn models are serialised, we can fully replicate the previous instance without
having to go through a class constructor. In other words, this and the ``from_checkpoint`` classmethod
that we will be implementing soon are functionally equivalent.
https://hub.docker.com/r/bentoml/model-server/tags
.. code-block:: python
    :dedent: 0

        def load(self, path):
            self.pipeline = joblib.load(path)

Note that DVC is taken care of by KTT at the pulling phase - your model need only push it.

``from_checkpoint``
^^^^^^^^^^^^^^^^^^^

This is a ``@classmethod`` to be used as an alternative constructor to ``__init__()``. It will be capable of fully reading the checkpoint to construct an exact replica of the model by itself, topology included, without needing the user to input the correct hierarchical metadata. Or that's what applied to PyTorch models.

For Scikit-learn models, again the checkpoint already contains the code. In other words, we can just create a blank instance then call its ``load`` method on the checkpoint! This is possible thanks to the workaround above.

.. code-block:: python
    :dedent: 0

        @classmethod
        def from_checkpoint(cls, path):
            instance = cls()
            cls.load(path)
            return instance

Doing it this way allows us to reuse the DVC handling implemented in ``cls.load()``.

``get_preprocessor``
^^^^^^^^^^^^^^^^^^^^
For optimum performance with tf-idf vectorisers, we will stem the words before passing them to this model. KTT provides a preprocessor for this, called ``SnowballStemmerPreprocessor``, which as its name suggests, borrows NLTK's SnowballStemmer facilities.

.. code-block:: python
    :dedent: 0

		@classmethod
		def get_preprocessor(cls, config):
		    """Return a SnowballStemmere instance for this model."""
		    return SnowballStemmerPreprocessor(config)

``fit``
^^^^^^^

Every model in KTT knows how to train itself, the process of which is implemented as the ``fit`` method. For sklearn models, we take in a training set (as returned by ``model_sklearn.get_loaders``), iterate over them for a set number of epochs, compute loss value and backpropagate the layers. Since every model is different in their training process (such as different loss functions, optimisers and such), it makes more sense to pack the training process into the models themselves.

Sklearn's high-level design shines again here, with the ``fit`` method being super short compared to PyTorch implementations:

.. code-block:: python
    :dedent: 0

        def fit(
                self,
                train_loader,
                val_loader=None,  # Unused but included for signature compatibility
                path=None,
                best_path=None,
                dvc=True
        ):
            self.pipeline.fit(train_loader[0], train_loader[1])
            if path is not None or best_path is not None:
                # There's no distinction between path and best_path as there is
                # no validation phase.
                self.save(path if path is not None else best_path, dvc)
            return None

``test``
^^^^^^^^

This method simply iterates the model over any given dataset (usually the test set) as presented above. Since it will most likely be used for testing a newly-trained model against a test set, it's named ``test`` (quite creatively). It is pretty much a slightly adjusted copy of the validation logic found in ``fit``, so there's not much to go about.

The only thing of note is the output format. **All Scikit-learn-based KTT models' test methods are required to output a dictionary with at least four keys.** The first one, ``targets``, leads to the labels column. The second one, ``predictions``, contains the model's selected class names to be compared against ``targets``. The third one, ``targets_b``, is the same as the ``targets`` column but binarised (this can be easily done using sklearn's own facilities). The last one is ``scores``, which are the raw scores from the model before being argmaxed and matched back to label names.

In this implementation, we'll also output a fifth key, called ``encodings``. As we do not have a separate ``forward_with_features`` method as in the example PyTorch model, we chose to include such functionality into this method. Also, we will manually implement it here instead of using Pytorch's AvgPool layers, just to keep things exclusively sklearn and ``numpy``.

.. code-block:: python
    :dedent: 0

        def test(self, loader, return_encodings=False):
		    # We need binarised targets for AU(PRC)
			y_avg = preprocessing.label_binarize(
		        loader[1],
		        classes=self.pipeline.classes_
		    )
		    # Separately run each stage so we can extract the feature vectors
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


``gen_reference_set``
^^^^^^^^^^^^^^^^^^^^^

This is where we generate the reference dataset for production-time model performance monitoring.

Our goal is to create a Pandas dataframe with the columns detailed in :ref:`reference-set`, that is, one column for every feature (titled with a stringified number starting from 0), then one column for every leaf label's classification score (titled with the label names).

There's a catch, however: Since this model runs without using the JSON, it only knows of internal indices instead of textual label names. In other words, we will have name collisions (against the feature column names, which are also numbers). To circumvent this, we spice up the terminology by prepending some letters to these names. 'C' for labels and 'F' for features should work well.

.. code-block:: python
    :dedent: 0

        def gen_reference_set(self, loader):
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


As you can see, this method is very similar to the ``test`` method above - in fact, it calls ``test`` to get most the necessary data. It additionally pools and stores features since we shouldn't be tracking a ton of separate columns at once - too much overhead for too little gain in insight.

Implementing the BentoService
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's take a break from ``testmodel.py`` and focus on implementing the actual BentoService that will run our model. In other words, let's move to ``bentoml/svc_lts.py``.
Each model will have differing needs for pre- and post-processing as well as metadata and data flow. Due to this, we have decided to let each model implement their own BentoService runtime.

 As of BentoML LTS 0.13, ONNX is supported but rather buggy for those who want to use GPUs for inference. As such, in this guide we will instead simply serialise our components and then load them into the BentoService runtime. This has the added benefit of having almost identical code between BentoService and the ``test`` method.

First, we import all the dependencies needed at inference time and read a few environment variables. This will involve a bunch of BentoML modules, which are very well explained in `their official documentation <https://docs.bentoml.org/en/0.13-lts/>`_.

.. code-block:: python

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
	SNOWBALLSTEMMER = SnowballStemmer('english')
	STOP_WORDS = set(stopwords.words('english'))

	EVIDENTLY_HOST = os.environ.get('EVIDENTLY_HOST', 'localhost')
	EVIDENTLY_PORT = os.environ.get('EVIDENTLY_PORT', 5001)

	REFERENCE_SET_FEATURE_POOL = 64

Note the two environment variables here (``EVIDENTLY_HOST`` and ``EVIDENTLY_PORT``). This is to allow the different components of our service to be run both directly on host machine's network as well as being containerised in a Docker network (in which hostnames are not just ``localhost`` anymore). KTT will provide the necessary ``docker-compose`` configuration to set these environment variables to the suitable values, so reading them here and using them correctly is really all we need to do.

Next, we need to implement the service class. It will be a subclass of ``bentoml.BentoService``. All of its dependencies, data (called 'artifacts') and configuration are defined via @decorators, as BentoML internally uses a dependency injection framework.

.. code-block:: python

	@bentoml.env(
		requirements_txt_file='models/db_bhcn/bentoml/requirements.txt'
	)
	@bentoml.artifacts([
		SklearnModelArtifact('model'),
		JSONArtifact('config'),
	])
	class TestModel(bentoml.BentoService):
		"""Real-time inference service for TestModel."""

		_initialised = False

		def init_fields(self):
		    """Initialise the necessary fields. This is not a constructor."""
		    self.model = self.artifacts.model
		    # Load service configuration JSON
		    self.monitoring_enabled = self.artifacts.config['monitoring_enabled']
		    self.pooled_feature_size = self.model.n_features_in_ // REFERENCE_SET_FEATURE_POOL

		    self._initialised = True
		    
Lastly, we implement the actual predict() API handler as a method in that class, wrapped by a ``@bentoml.api`` decorator that defines the input type (for informing the outer BentoML web server) and microbatching specification.

.. code-block:: python
	:dedent: 0

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
		        ' '.join([SNOWBALLSTEMMER.stem(word) if word not in STOP_WORDS
		                  else word for word in lst])
		        for lst in tokenized
		    ]
		    tfidf_encoding = self.model.steps[0].transform(stemmed)
		    scores = self.model.steps[1].steppredict_proba(tfidf_encoding)
		    predictions = [
		        self.model.classes_[i] for i in np.argmax(scores, axis=1)]
		    
There's one more thing in this method to implement: some code to send the newly-received data-in-the-wild plus our model's scores for it to the monitoring service.
For more information regarding the format of the data to be sent to the monitoring service, please see :ref:`service-spec`.

.. code-block:: python
    :dedent: 0

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
		            (len(stemmed), 1 + self.pooled_feature_size + len(self.pipeline.classes_)),
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
		            for i in range(0, self.pooled_feature_size)
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
		        
Lastly, return the predictions. There is no need to post-process - Scikit-learn models do that by themselves and return the class names as discovered from the datasets!

.. code-block:: python
	:dedent: 0

		    return predictions

The configuration files
^^^^^^^^^^^^^^^^^^^^^^^

It's time to populate two out of the three configuration files in the ``./bentoml`` directory.

For ``evidently.yaml``, follow the guide at :ref:`bentoml-config`. Here's what you should end up with:

.. code-block:: yaml

    service:
        reference_path: './references.parquet'
        min_reference_size: 30
        use_reference: true
        moving_reference: false
        window_size: 30
        calculation_period_sec: 60
        monitors:
            - cat_target_drift
            - data_drift

For ``requirements.txt``, you should manually skim over your implementation and decide on which dependency will be needed at inference time (note: you don't need to include dependencies that are only used for training for obvious reasons). For this ``testmodel``, you might get the following:

.. code-block::

   bentoml==0.13.1
   scikit-learn==0.24.2
   numpy==1.19.5

It is always good practice to lock your versions. Only manually update a dependency version when necessary. This prevents breakages, as big Python libraries are known to fight each other over their own dependencies' versions.

For ``dashboard.json``, simply leave it blank for now.

``export``
^^^^^^^^^^

We will implement both export schemes: ONNX and BentoML.

Exporting to ONNX is relatively straightforward if not for the fact that transformer models need to be dealt with specially. For this reason, we export the DistilBERT encoder and the classifier head as separate ONNX graphs using different facilities.

For more information regarding naming and path specifications, please see :ref:`model-export-general`.

.. code-block:: python
    :dedent: 0

		def export_onnx(self, classifier_path, encoder_path=None):
		    initial_type = [('str_input', StringTensorType([None, 1]))]
		    onx = to_onnx(
		        self.pipeline, initial_types=initial_type, target_opset=11
		    )
		    # Export
		    with open(classifier_path + 'classifier.onnx', "wb") as f:
		        f.write(onx.SerializeToString())

Exporting as a BentoService is a bit more involved. We will implement it to support an optional monitoring extension powered by the Evidently library. This will be run as a standalone server accepting new data from production to compare with the above reference dataset to compute feature and target drift. To ease this process, KTT has already implemented said standalone server to be customisable (meaning new models can simply write a configuration file to tailor it to their needs and capabilities), as well as automating the file and folder logic for you. All you need to do is to produce two specific pieces of data: a configuration dictionary that lists out the features and classes this model has been trained on, and a fully packed BentoService instance.

We will now use the above facilities to export our new model as a self-contained, standalone classification service.

.. code-block:: python
    :dedent: 0

		def export_bento_resources(self, svc_config={}):
		    # Config for monitoring service
		    config = {
		        'prediction': self.classifier.hierarchy.classes[
		            self.classifier.hierarchy.level_offsets[-2]:
		            self.classifier.hierarchy.level_offsets[-1]
		        ]
		    }
		    svc = svc_lts.Tfidf_LSGD()
		    svc.pack('model', self.pipeline)
		    svc.pack('config', svc_config)

		    return config, svc


Registering, testing & conclusion
---------------------------------

With every part of your model implemented, now is the time to add it to the model list and implement some runner code to get the training and exporting script to use it smoothly. For this, you can refer to :ref:`model-register`.

Be sure to test out every option for your model before deploying to a production environment. Testing instructions can be found at :ref:`test-run`. Afterwards, design a Grafana dashboard and add it to the provisioning system to have your service automatically initialise Grafana right from the get-go.

After this, your model is pretty much complete. If you did it correctly, it should be an integral and uniform part of your own KTT fork and can be used just like any existing (bundled) model.


