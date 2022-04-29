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

The source implementation itself must subclass the abstract ``Model`` class (see :ref:`model-class`), like in any other framework.

Scikit-learn utilities
----------------------

KTT provides framework-specific utilities for common tasks such as loading data in and computing performance metrics. For sklearn, the following are available:

.. automodule:: models.model_sklearn
    :members:

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

In ``testmodel.py``, import the necessary libraries and define a concrete subclass of the ``Model`` abstract class:

.. code-block:: python

    import os
    import joblib

    from sklearn import preprocessing, linear_model
    from sklearn.pipeline import Pipeline

    from sklearn.feature_extraction.text import TfidfVectorizer
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import StringTensorType
    import bentoml

    from models import model


    class TestModel(model.Model):
        """A wrapper class around the scikit-learn-based test model.

        It's basically a replica of the Tfidf-LeafSGD model bundled with KTT.
        """

        def __init__(self, config=None, verbose=False):
            pass

        @classmethod
        def from_checkpoint(cls, path):
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

        def export(self, dataset_name, bento=False):
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

.. code-block:: python
    :dedent: 0
        def __init__(self, config=None, verbose=False):
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

.. code-block:: python
    :dedent: 0

        def load(self, path):
            if not os.path.exists(path):
                if not os.path.exists(path + '.dvc'):
                    raise OSError('Checkpoint not present and cannot be retrieved')
            os.system('dvc checkout {}.dvc'.format(path))
            self.pipeline = joblib.load(path)

Note that DVC integration is a required part of this function - there is no parameter to enable/disable it and as such the training script assumes that all ``load`` implementation handles DVC checkouts by themselves.

``from_checkpoint``
^^^^^^^^^^^^^^^^^^^

This is a ``@classmethod`` to be used as an alternative constructor to ``__init__()``. It will be capable of fully reading the checkpoint to construct an exact replica of the model by itself, topology included, without needing the user to input the correct hierarchical metadata. Or that's what applied to PyTorch models.

For Scikit-learn models, again the checkpoint already contains the code. In other words, we can just create a blank instance then call its ``load`` method on the checkpoint!

.. code-block:: python
    :dedent: 0

        @classmethod
        def from_checkpoint(cls, path):
            instance = cls()
            cls.load(path)
            return instance

Doing it this way allows us to reuse the DVC handling implemented in ``cls.load()``.

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
            y_avg = preprocessing.label_binarize(
                loader[1],
                classes=self.pipeline.classes_
            )
            tfidf_encoding = self.pipeline.steps[0].transform(loader[0])
            scores = self.pipeline.steps[1](tfidf_encoding)
            predictions = [self.pipeline.classes_[i] for i in np.argmax(scores, axis=1)]

            res = {
                'targets': loader[1],
                'targets_b': y_avg,
                'predictions': predictions,
                'scores': scores,
            }
            if return_encodings:
                # Average-pool encodings
                res['encodings'] = np.array([
                    np.average(
                        tfidf_encoding[
                            :,
                            i*REFERENCE_SET_FEATURE_POOL:
                            min((i+1)*REFERENCE_SET_FEATURE_POOL, len(scores))
                        ]
                    )
                    for i in range(0, pooled_feature_size)
                ])
            return res

``gen_reference_set``
^^^^^^^^^^^^^^^^^^^^^

This is where we generate the reference dataset for production-time model performance monitoring.

Our goal is to create a Pandas dataframe with the columns detailed in :ref:`reference-set`, that is, one column for every feature (titled with a stringified number starting from 0), then one column for every leaf label's classification score (titled with the label names).

.. code-block:: python
    :dedent: 0

        def gen_reference_set(self, loader):
            results = self.test(loader, return_encodings=True)
            pooled_features = results['encodings']
            scores = results['scores']
            pooled_feature_size = self.pipeline.n_features_in_ // REFERENCE_SET_FEATURE_POOL
            targets = loader[1]
            scores = result['scores']
            cols = {
                'targets': targets,
            }
            for col_idx in range(pooled_features[1]):
                cols[str(col_idx)] = pooled_"""Service file for Tfidf-LeafSGD."""
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
class DB_BHCN(bentoml.BentoService):
    """Real-time inference service for DB-BHCN."""

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
        return predictionsfeatures[:, col_idx]
            for col_idx in range(scores.shape[1]):
                cols[self.pipeline.classes_[col_idx]] = scores[:, col_idx]
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

Note the two environment variables here (``EVIDENTLY_HOST`` and ``EVIDENTLY_PORT``). This is to allow the different components of our service to be run both directly on host machine's network as well as being containerised in a Docker network (in which hostnames are not just ``localhost`` anymore). KTT will provide the necessary ``docker-compose`` configuration to set these environment variables to the suitable values, so reading them here and using them correctly is really all we need to do.

Next, we need to implement the service class. It will be a subclass of ``bentoml.BentoService``. All of its dependencies, data (called 'artifacts') and configuration are defined via @decorators, as BentoML internally uses a dependency injection framework.

.. code-block:: python

	@bentoml.env(
		requirements_txt_file='models/db_bhcn/bentoml/requirements.txt',
		docker_base_image='bentoml/model-server:0.13.1-py36-gpu'
	)
	@bentoml.artifacts([
		SklearnModelArtifact('model'),
		JSONArtifact('config'),
	])
	class DB_BHCN(bentoml.BentoService):
		"""Real-time inference service for DB-BHCN."""

		_initialised = False

		def init_fields(self):
		    """Initialise the necessary fields. This is not a constructor."""
		    self.model = self.artifacts.model
		    # Load service configuration JSON
		    self.monitoring_enabled = self.artifacts.config['monitoring_enabled']
		    self.pooled_feature_size = self.model.n_features_in_ // REFERENCE_SET_FEATURE_POOL

		    self._initialised = True
		    
Lastly, we implement the actual predict() API handler as a method in that class, wrapped by a ``@bentoml.api`` decorator that defines the input type (for informing the outer BentoML web server) and microbatching specification.

..code-block:: python

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

        def export(
                self, dataset_name, bento=False, reference_set_path=None
        ):
            self.eval()
            # Create dummy input for tracing
            batch_size = 1  # Dummy batch size. When exported, it will be dynamic
            x = torch.randn(batch_size, 768, requires_grad=True).to(
                self.device
            )
            if not bento:
                    # Export to ONNX graphs.
                    # KTT provides utilities for exporting trained DistilBERT instances.
                export_trained(
                    self.encoder,
                    dataset_name,
                    'testmodel',
                )
                # Prepare names and paths
                name = '{}_{}'.format(
                    'testmodel',
                    dataset_name
                )
                path = 'output/{}/classifier/'.format(name)
                if not os.path.exists(path):
                    os.makedirs(path)
                path += 'classifier.onnx'
                # Clear previous versions
                if os.path.exists(path):
                    os.remove(path)
                # Export into transformers model .bin format
                # Since our model is minibatched, we have to make the first axis
                # dynamic. In production, batch sizes can vary depending on load
                # (or stay at 1 if no microbatching is available).
                torch.onnx.export(
                    self.classifier,
                    x,
                    path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                # Additionally export hierarchical metadata to the same folder.
                self.classifier.hierarchy.to_json(
                    "output/{}/hierarchy.json".format(name)
                )

Exporting as a BentoService is more involved. We need to implement it to support an optional monitoring extension powered by the Evidently library. This will be run as a standalone server accepting new data from production to compare with the above reference dataset to compute feature and target drift. To ease this process, KTT has already implemented said standalone server to be customisable (meaning new models can simply write a configuration file to tailor it to their needs and capabilities), as well as a utility function for generating the resulting service folders and configuration files:

.. autoclass:: utils.build.init_folder_structure
    :noindex:

We will now use the above facilities to export our new model as a self-contained, standalone classification service.

.. code-block:: python
    :dedent: 0

            else:
                # Export as BentoML service
                build_path = 'build/testmodel_' + dataset_name.lower()
                build_path_inference = ''
                if reference_set_path is not None:
                        # If a path to a reference dataset is available, export the
                        # model as a service with monitoring capabilities.
                    with open(
                            'models/testmodel/bentoml/evidently.yaml', 'r'
                    ) as evidently_template:
                        config = yaml.safe_load(evidently_template)
                        config['prediction'] = self.classifier.hierarchy.classes[
                            self.classifier.hierarchy.level_offsets[-2]:
                            self.classifier.hierarchy.level_offsets[-1]
                        ]
                    # Init folder structure, Evidently YAML and so on.
                    build_path_inference = init_folder_structure(
                        build_path,
                        {
                            'reference_set_path': reference_set_path,
                            'grafana_dashboard_path':
                                'models/testmodel/bentoml/dashboard.json',
                            'evidently_config': config
                        }
                    )
                else:
                        # Init folder structure for a minimum system (no monitoring)
                    build_path_inference = init_folder_structure(build_path)
                # Initialise a BentoService instance - we'll come to this soon
                svc = svc_lts.TestModel()
                # Pack tokeniser along with encoder. Here we use KTT's DistilBERT
                # facilities.
                encoder = {
                    'tokenizer': get_tokenizer(),
                    'model': self.encoder
                }
                svc.pack('encoder', encoder)
                svc.pack('classifier', torch.jit.trace(self.classifier, x))
                svc.pack('hierarchy', self.classifier.hierarchy.to_dict())
                svc.pack('config', {
                    'monitoring_enabled': reference_set_path is not None
                })
                # Export the BentoService to the correct path.
                svc.save_to_dir(build_path_inference)

Registering, testing & conclusion
---------------------------------

With every part of your model implemented, now is the time to add it to the model list and implement some runner code to get the training and exporting script to use it smoothly. For this, you can refer to :ref:`model-register`.

Adding the model to the training script is quite simple. You can follow implementations for bundled models and adapt them to your own. Below is a sample implementation:

.. code-block:: python

    if 'testmodel' in model_lst:
    # If your model has hyperparameters in ./hyperparameters.json:
    # config = init_config('testmodel', 'Test Model')
    TestModel = __import__('models', globals(), locals(), [], 0).TestModel
    for dataset_name in dataset_lst:
        (
            train_loader, val_loader, test_loader, hierarchy, config
        ) = init_dataset(
            dataset_name, model_pytorch.get_loaders, config
        )
        model = TestModel(hierarchy, config).to(device)
        train_and_test(
            config,
            model,
            train_loader,
            val_loader,
            test_loader,
            metrics_func=model_pytorch.get_metrics,
            dry_run=dry_run,
            verbose=verbose,
            gen_reference=reference,
            dvc=dvc
        )

Similarly for the export script:

.. code-block:: python

    if 'testmodel' in model_lst:
        click.echo('{}Exporting {}...{}'.format(
            cli.BOLD, 'Test model', cli.PLAIN))
        ModelClass = __import__(
            'models', globals(), locals(), [], 0).TestModel
        model = TestModel.from_checkpoint(
            get_path('testmodel', dataset_name, best=best, time=time),
        ).to(device)
        if monitoring:
            reference_set_path = get_path(
                'testmodel', dataset_name, time=time, reference_set=True)
            if reference_set_path is not None:
                model.export(dataset_name, bento, reference_set_path)
            else:
                model.export(dataset_name, bento)

Be sure to test out every option for your model before deploying to a production environment. Testing instructions can be found at :ref:`test-run`. Afterwards, design a Grafana dashboard and add it to the provisioning system to have your service automatically initialise Grafana right from the get-go.

After this, your model is pretty much complete. If you did it correctly, it should be an integral and uniform part of your own KTT fork and can be used just like any existing (bundled) model.


