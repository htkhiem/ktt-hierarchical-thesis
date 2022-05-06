.. Dev section - developing with PyTorch.

Implementing a model with PyTorch+DistilBERT
============================================

PyTorch is our framework of choice when it comes to massively-parallelised models such as neural networks. All PyTorch models currently use DistilBERT as their text-to-number encoder. Support for other encoders will be added in the future.

This article will guide you through the process of implementing a model from scratch using KTT's facilities and PyTorch. It will mainly focus on standout aspects (those that are unique to PyTorch).

The model
---------

To keep everything simple and focus on the system integration aspects only, we will recreate DistilBERT+Linear, which is the simplest of all DistilBERT/PyTorch-based models. It consists of a DistilBERT tokeniser+encoder which feeds features to a single linear layer as the classifier head. More information about this model can be found in :ref:`db-linear-theory`.


PyTorch model module structure
------------------------------

Each PyTorch model module ('module' for short) in KTT is a self-contained collection of implemented source code, metadata and configuration files. A module defines its own training, checkpointing and exporting procedures. It might also optionally implement a  BentoML service and configuration files for live inference using the integrated BentoML-powered inference system and monitoring using Prometheus/Grafana. The general folder tree of a PyTorch model is as detailed in :ref:`model-struct`.

The source implementation itself must subclass the abstract ``PyTorchModel`` class, which subclasses the abstract ``Model`` class (see :ref:`model-class`) pre-implements two of the abstract methods for you (``get_dataloader_func`` and ``get_metrics_func``). PyTorch models with additional submodules (bundled example: DB-BHCN and its AWX submodule, or DistilBERT+Adapted C-HMCNN with its ``MCM`` submodule) must implement a ``to(self, device)`` method similar to PyTorch's namesake method to recursively transfer the entire instance to the specified device.

PyTorch utilities
-----------------

KTT provides framework-specific utilities for common tasks such as loading data in and computing performance metrics. For PyTorch, the following are available:

.. automodule:: models.model_pytorch
    :members:
    :exclude-members: CustomDataset
    
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

- ``encoder_lr``: Encoder (DistilBERT) learning rate.
- ``classifier_lr``: Classifier head learning rate.
- ``dropout``: The dropout probability for that dropout layer between DistilBERT and the linear layer in our classifier head.

In addition to those, all PyTorch models must also define:

- ``train_minibatch_size``: Minibatch size in the training phase, for passing to ``get_loaders()`` in the common training script (``train.py``).
- ``val_test_minibatch_size``: Similarly for validation and test phase.
- ``model_name``: A convenience field to help with file management. Simply set this to ``testmodel``.

At least our model-specific hyperparameters will have to be present in the ``config`` dict that we will soon see in the upcoming parts.

Implementing the model
~~~~~~~~~~~~~~~~~~~~~~

From here on we will refer to files using their paths in relative to the ``testmodel`` folder.

In ``testmodel.py``, import the necessary libraries and define a concrete subclass of the ``PyTorchModel`` abstract class:

.. code-block:: python

    """Implementation of our toy test model."""
	import os
	import pandas as pd
	import torch
	import numpy as np
	from tqdm import tqdm

	from models import model_pytorch
	from utils.hierarchy import PerLevelHierarchy
	from utils.encoders.distilbert import get_pretrained, get_tokenizer, \
		export_trained, DistilBertPreprocessor
	from .bentoml import svc_lts

    class TestModel(model_pytorch.PyTorchModel, torch.nn.Module):
        """Wrapper class combining DistilBERT with a linear model."""

        def __init__(
            self,
            hierarchy,
            config  # The config dict mentioned above here
        ):
            pass

        @classmethod
        def from_checkpoint(cls, path):
            pass
            
        @classmethod
        def get_preprocessor(cls, path):
            pass

        def forward(self, ids, mask):
            pass

        def forward_with_features(self, ids, mask):
            pass

        def save(self, path, optim, dvc=True):
            pass

        def load(self, path):
            pass

        def fit(
                self,
                train_loader,
                val_loader,
                path=None,
                best_path=None,
                resume_from=None,
                dvc=True
        ):
            pass

        def test(self, loader):
            pass

        def gen_reference_set(self, loader):
            pass

		def export_onnx(self, classifier_path, encoder_path):
			pass
			
		def export_bento_resources(self, svc_config={}):
			pass

        def to(self, device=None):
            pass

    if __name__ == "__main__":
        pass

You might notice that there are more methods than what is there in the ``Model`` abstract class. Some of them are PyTorch-specific, while others are for reference dataset generation. Since we do not force every model to be able to export to our BentoML-based inference system with full monitoring capabilities, these methods are not defined in the abstract class. However, they will be covered in this guide for the sake of completeness.

Now we will go through the process of implementing each method.

.. note::

    We highly recommend writing documentation for your model as you implement each method.

    KTT's documentation system uses Sphinx but follows PEP 8's documentation strings standard, with Sphinx features exposed to the syntax via the ``numpydoc`` extension. In short, you can refer to `this style guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

    The below code listings will not include full documentation (only short summary docstrings) for brevity.

``__init__``
^^^^^^^^^^^^

Constructing a PyTorch model involves initialising all the submodules, and our framework is no exception. In addition to that, we also save a bit of hierarchy-related metadata along for easier access during the exporting process.

There are a few design details that might need further elaboration, all of which are written as comments in the code block below. Do take time to read through them to gain a better understanding of how and why we do things that way.

.. code-block:: python
    :dedent: 0

        def __init__(
            self,
            hierarchy,
            config
        ):
            """Construct module."""

            # PyTorch module init ritual
            super(TestModel, self).__init__()
            # All models default to CPU processing. This is to stay consistent
            # with PyTorch's builtin modules. You can easily move an instance to
            # another device by using self.to(device) later.
            self.device = 'cpu'

            # DistilBERT
            # This utility function returns a fresh pretrained instance
            # of DistilBERT. The tokeniser on the other hand is already built
            # into the dataset loader for all PyTorch models.
            self.encoder = get_pretrained()

            # Our classifier head is simply a dropout layer followed by a single
            # linear layer
            self.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(p=config['dropout']),  # read the dropout hyperparam here
                    torch.nn.Linear(input_dim, hierarchy.levels[-1])  # output size = leaf layer size
            )
            # Back these up for checkpointing and exporting
            self.config = config
            self.hierarchy = hierarchy
            self.output_size = hierarchy.levels[-1]

            # We'll talk about this later
            self.pool = torch.nn.AvgPool1d(32)
        
Note the pooling layer at the end. It has been set up with a kernel size (and stride size too, by default) of 32. When applied on the 768 features DistilBERT gives us, we will get 768/32=24 average-pooled features. This will come in handy later for the monitoring system implementation part, so stay tuned.

``save``
^^^^^^^^

Let's finish this method first to freeze our checkpoint schema before we implement ``load`` and ``from_checkpoint``.

Checkpoint format and schema in KTT are again dependent on the implementation of the model. There is no rigid design, but there are requirements that all designs must fulfill:

- The checkpoint must contain sufficient data to fully replicate the instance that produced the checkpoint. In our case, this means we have to additionally include all the data passed to the constructor, which is the hierarchy and the configuration dict, and also the optimiser's state, which allows us to later resume training from the last epoch that instance was trained on.
- The checkpoint must be a single file. If you must have multiple files, please build an archive.

Additionally, our checkpoint design should attempt to be consistent with those produced by existing (bundled) models. For this, we will pack all checkpoint data into a dict, then use Python pickling to serialise it. However, since we are storing model weights as PyTorch tensors, we cannot directly use the normal pickle library. Instead, we use PyTorch's ``load`` and ``save`` utilities to correctly deal with the underlying storage of those tensors.

.. code-block:: python
    :dedent: 0

        def save(self, path, optim, dvc=True):
            checkpoint = {
                'config': self.config,
                'hierarchy': self.classifier.hierarchy.to_dict(),
                'encoder_state_dict': self.encoder.state_dict(),
                'classifier_state_dict': self.classifier.state_dict(),
                'optimizer_state_dict': optim
            }
            torch.save(checkpoint, path)

            if dvc:
                os.system('dvc add ' + path)

Note that we also automate the process of adding the checkpoint to DVC tracking, which is controlled by the ``dvc`` flag. The schema of the checkpoint dict is also readily visible in the code listing.

``load``
^^^^^^^^

This is the reverse of ``save``. Simply use PyTorch's ``load`` function to load our previously-saved checkpoint back into this instance. However, the current instance must be similar to the past one in the hierarchy they have been instantiated to - in other words, a checkpoint loaded this way must be from a model with the exact same layer sizes. For replicating an instance from scratch, including its layer sizes, use the ``from_checkpoint`` alternative constructor instead.

.. code-block:: python
    :dedent: 0

        def load(self, path):
                # DVC automation: if checkpoint file is not found, see if
                # it's tracked with DVC
            if not os.path.exists(path):
                    # DVC tracking placeholder does not exist
                if not os.path.exists(path + '.dvc'):
                    raise OSError('Checkpoint not present and cannot be retrieved')
                # DVC tracking placeholder exists. Retrieve the file from remote.
                os.system('dvc checkout {}.dvc'.format(path))
            # Now that we have ensured that the file exists locally, load it in.
            checkpoint = torch.load(path)
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
            # Return the optimiser state dict so the fit() function can do its job.
            return checkpoint['optimizer_state_dict']
        
Note that DVC integration is a required part of this function - there is no parameter to enable/disable it and as such the training script assumes that all ``load`` implementation handles DVC checkouts by themselves.
        
``from_checkpoint``
^^^^^^^^^^^^^^^^^^^

This is a ``@classmethod`` to be used as an alternative constructor to ``__init__()``. It will be capable of fully reading the checkpoint to construct an exact replica of the model by itself, topology included, without needing the user to input the correct hierarchical metadata.

.. code-block:: python
    :dedent: 0

        @classmethod
        def from_checkpoint(cls, path):
            if not os.path.exists(path):
                # DVC tracking placeholder does not exist
                if not os.path.exists(path + '.dvc'):
                    raise OSError('Checkpoint not present and cannot be retrieved')
                # DVC tracking placeholder exists. Retrieve the file from remote.
                os.system('dvc checkout {}.dvc'.format(path))
            # Now that we have ensured that the file exists locally, load it in.
            checkpoint = torch.load(path)
            # Where this differs from self.load(): it constructs a new instance instead
            # of loading the checkpoint into an existing instance.
            hierarchy = PerLevelHierarchy.from_dict(checkpoint['hierarchy'])
            instance = cls(hierarchy, checkpoint['config'])
            # From this part onwards it is pretty much identical to self.load().
            # You might instead call instance.load(path).
            instance.encoder.load_state_dict(
                checkpoint['encoder_state_dict']
            )
            instance.classifier.load_state_dict(
                checkpoint['classifier_state_dict']
            )
            return instance
        
Again, DVC handling is assumed to be part of your implementation. It should not differ from the ``load`` function in this regard, so you might as well copy it over, or refactor into a separate private method.

``get_preprocessor``
^^^^^^^^^^^^^^^^^^^^
Our test model uses DistilBERT for encoding, so we need to preprocess the incoming text to suit its needs. Specifically, we need to use the same preprocessing and tokenisation DistilBERT was trained upon, which KTT has wrapped in a ``BasePreprocessor`` subclass called ``DistilBertPreprocessor``:

.. code-block:: python
	:dedent: 0

		@classmethod
		def get_preprocessor(cls, config):
		    """Return a DistilBERT preprocessor instance for this model."""
		    return DistilBertPreprocessor(config)
	
``forward``
^^^^^^^^^^^

The core of every PyTorch model is the ``forward()`` method. Similarly to PyTorch modules, this method implements how input data flows through the topology and become output data to be returned.

Our simple model will simply accept ``(ids, mask)`` as returned by the DistilBERT tokeniser (more on this in the ``fit()`` method), send it to the local DistilBERT encoder instance then forward the last hidden layer's ``[CLS]`` token to the classifier head. The output of the classifier head is a tensor of classification scores at the leaf level, of shape (minibatch, level_sizes[-1]).

.. code-block:: python
    :dedent: 0

        def forward(self, ids, mask):
            return self.classifier(
                self.encoder(
                    ids, attention_mask=mask
                )[0][:, 0, :]
            )
        
``fit``
^^^^^^^

Every model in KTT knows how to train itself, the process of which is implemented as the ``fit`` method. Here we take in a training set and a validation set (both packaged as minibatched and shuffled PyTorch DataLoaders), iterate over them for a set number of epochs, compute loss value and backpropagate the layers. Since every model is different in their training process (such as different loss functions, optimisers and such), it makes more sense to pack the training process into the models themselves.

This is arguably the longest out of all methods, so we will present it in part instead of in whole. The first part involves setting up the loss function, optimiser and some related information before any training could begin.

.. code-block:: python
    :dedent: 0

        def fit(
                self,
                train_loader,
                val_loader,
                path=None,
                best_path=None,
                resume_from=None,
                dvc=True
        ):
            # Keep min validation (test set) loss so we can separately back up our
            # best-yet model
            val_loss_min = np.Inf
            # Initialise the loss function. For this model, we will use a simple
            # CrossEntropyLoss. It is the general case of the more common BCELoss.
            criterion = torch.nn.CrossEntropyLoss()
            # Backpropagation learning rates will be handled by a typical Adam
            # optimiser. Note how we allow different learning rates for DistilBERT
            # and the classifier head. This allows more flexibility in avoiding
            # catastrophic forgetting.
            optimizer = torch.optim.Adam([
                {
                    'params': self.encoder.parameters(),
                    'lr': self.config['encoder_lr']
                },
                {
                    'params': self.classifier.parameters(),
                    'lr': self.config['classifier_lr']
                }
            ])
            # Store validation metrics after each epoch
            val_metrics = np.empty((4, 0), dtype=float)

After initialising them all, the training phase could begin. A DataLoader can be seen as a list of minibatches, whose order are configured to be shuffled every time an iterable is requested. The size of the minibatch will be configured somewhere else (not within this model's scope).

Each minibatch produced by a PyTorch DataLoader in KTT's PyTorch framework is a dictionary with the following fields:

- ``ids``: the token ID tensor, computed by a DistilBERT tokeniser. All strings are padded or truncated to 512 tokens by default.
- ``mask``: DistilBERT's attention mask input, also from the same tokeniser.
- ``targets``: the target label index tensor, of shape (minibatch, depth). Each row represents the targets, in hierarchical order, for a single example in the minibatch.
- (optionally) ``targets_b``: Like ``targets``, but binarised using the above ``get_hierarchical_one_hot`` utility function.

.. code-block:: python
    :dedent: 0

            # Loop for each training epoch. Note how we use the 'epoch' field in
            # the hyperparameters config dict.
            for epoch in range(1, self.config['epoch'] + 1):
                    # Keep track of this epoch's loss accumulated validation loss so we can
                    # compare this epoch with the best-performing one.
                val_loss = 0
                    # Set the model to training mode. This is needed due to us inheriting
                    # PyTorch's Module class.
                self.train()
                for batch_idx, data in enumerate(tqdm(train_loader)):
                        # Extract the necessary fields from the minibatch dict
                    ids = data['ids'].to(self.device, dtype=torch.long)
                    mask = data['mask'].to(self.device, dtype=torch.long)
                    targets = data['labels'].to(self.device, dtype=torch.long)
                    # Use the just-implemented ``forward`` method to forward-propagate
                    # the minibatch.
                    output = self.forward(ids, mask)
                    # Clear accumulated gradients from the optimiser.
                    optimizer.zero_grad()
                    # Compute loss using our initialised loss function.
                    # This is a leaf-level model, so it only outputs
                    # classifications for the leaves.
                    # Similarly, we have to extract just the leaf targets
                    # (the last column).
                    loss = criterion(output, targets[:, -1])
                    # Back-propagate the loss value and iterate the optimiser.
                    loss.backward()
                    optimizer.step()

For every epoch, in addition to the training phase, we also perform a validation phase. This phase does not compute derivatives for backward propagation, so be sure to wrap it in a ``torch.no_grad()`` environment to both improve performance and to remove any chance of accidental training on the validation set. The rest of the code is quite similar to the training phase, except with the notable omission of back-propagation and the addition of metrics computation.

.. note::

    Since this is a leaf-only model (meaning it only classifies at the leaf level and does not benefit from hierarchical structures), its outputs and targets have one fewer dimension than true hierarchical models. To be specific, while true hierarchical models such as DB-BHCN will return their outputs with shape (example, level, level labels), our model only returns (example, leaf labels). Do not add a singleton dimension to this, as KTT's metrics facilities will handle leaf-only models separately.

.. code-block:: python
    :dedent: 0

                # Switch to the validation phase for this epoch.
                self.eval()
                # Keep track of all model outputs and the corresponding targets
                # for computing validation metrics in addition to the loss
                # functions
                val_targets = np.array([], dtype=float)
                val_outputs = np.empty((0, self.output_size), dtype=float)
                # Disable gradient descent for validation phase.
                with torch.no_grad():
                    for batch_idx, data in tqdm(enumerate(val_loader)):
                        ids = data['ids'].to(self.device, dtype=torch.long)
                        mask = data['mask'].to(self.device, dtype=torch.long)
                        targets = data['labels'].to(self.device, dtype=torch.long)
                        output = self.forward(ids, mask)
                        loss = criterion(output, targets[:, -1])

                        # Record model outputs and corresponding targets
                        val_targets = np.concatenate([
                            val_targets, targets.cpu().detach().numpy()[:, -1]
                        ])
                        val_outputs = np.concatenate([
                            val_outputs, output.cpu().detach().numpy()
                        ])
                    # Compute metrics on this minibatch
                    val_metrics = np.concatenate([
                        val_metrics,
                        # get_metrics returns a 1D array so we have to add
                        # another dimension before we can concatenate it to
                        # the val_metrics array.
                        np.expand_dims(
                            get_metrics(
                                {
                                    'outputs': val_outputs,
                                    'targets': val_targets
                                },
                                display=None),
                            axis=1
                        )
                    ], axis=1)
                    # Create a checkpoint.
                    if path is not None and best_path is not None:
                        optim = optimizer.state_dict()
                        self.save(path, optim, dvc)
                            # If this is the new best-performing epoch, make an
                        # additional copy.
                        if val_loss <= val_loss_min:
                            print('Validation loss decreased ({:.6f} --> {:.6f}).'
                                  'Saving best model...'.format(
                                      val_loss_min, val_loss))
                            val_loss_min = val_loss
                            self.save(best_path, optim)
                    print('Epoch {}: Done\n'.format(epoch))

            # Return validation metrics of each epoch for external usage, such as
            # graphing performance over epochs.
            return val_metrics
        
``test``
^^^^^^^^

This method simply iterates the model over a DataLoader as presented above. Since it will most likely be used for testing a newly-trained model against a test set, it's named ``test`` (quite creatively). It is pretty much a slightly adjusted copy of the validation logic found in ``fit``, so there's not much to go about.

The only thing of note is the output format. **All PyTorch-based KTT models' test methods are required to output a dictionary with two keys.** The first one, ``targets``, contains all targets as iterated over the dataset, concatenated together into one long 2D array just like ``val_targets`` above. The second one, ``outputs``, contains the concatenated model outputs, again just like ``val_outputs`` above.

.. code-block:: python
    :dedent: 0

        def test(self, loader):
            self.eval()

            all_targets = np.array([], dtype=bool)
            all_outputs = np.empty((0, self.output_size), dtype=float)

            with torch.no_grad():
                for batch_idx, data in enumerate(tqdm(loader)):
                    ids = data['ids'].to(self.device, dtype=torch.long)
                    mask = data['mask'].to(self.device, dtype=torch.long)
                    targets = data['labels']

                    output = self.forward(ids, mask)

                    all_targets = np.concatenate([
                        all_targets, targets.numpy()[:, -1]
                    ])
                    all_outputs = np.concatenate([
                        all_outputs, output.cpu().detach().numpy()
                    ])
            return {
                'targets': all_targets,
                'outputs': all_outputs,
            }

``forward_with_features``
^^^^^^^^^^^^^^^^^^^^^^^^^

From this point onwards, we will deal with methods that facilitate exporting. KTT has built-in facilities for setting up self-contained, deployable classification services using BentoML. It also has presets for integrating monitoring capabilities provided by Evidently to get live statistics on how well the model is performing in a production environment.

This method implements the foundation to an important prerequisite to the monitoring aspect: a reference dataset. This dataset contains numerical features and the corresponding classification scores to detect feature and target drift. For our ``testmodel``, we will use DistilBERT encoder outputs as the numerical features. This requires us to have some way to return such encodings so we could log them down along with the output scores. Simply setting up a boolean flag to adjust what kind of value to return from the ``forward`` method will incur conditional branching on every minibatch and complicate what should essentially be a straightforward view of the data flow through the model. Instead, we will have a special version of ``forward`` that will only be called for generating this reference dataset and nothing else.

This is also where we use the average-pool layer instantiated way back in the ``__init__()`` constructor! So, why? The reason is due to how computationally-intensive the feature drift detection is. With 768 values to track, the monitoring feature is going to add a lot of overhead to the process even if it is only periodically run. This means some requests will be strangely slow, delaying your entire production system. Furthermore, there is no need to visualise the drift intensity of 768 values - which would simply clutter the heatmap and give us an unnecessarily detailed view to the situation. As such, by pooling 768 features into just 24, we effectively 'reduce' the resolution of the heatmap while still retaining a good ability to detect drifts early.

.. code-block:: python
    :dedent: 0

        def forward_with_features(self, ids, mask):
            encoder_outputs = self.encoder(ids, mask)[0][:, 0, :]
            local_outputs = self.classifier(
                encoder_outputs
            )
            # Remember to pool features here before returning!
            return local_outputs, self.pool(encoder_outputs)

``gen_reference_set``
^^^^^^^^^^^^^^^^^^^^^

This is where we use the above ``forward_with_features`` method to iterate over an input dataset and generate the corresponding dataset. Again, the input dataset will be wrapped in a minibatched DataLoader.

Our goal is to create a Pandas dataframe with the columns detailed in :ref:`reference-set`, that is, one column for every feature (titled with a stringified number starting from 0), then one column for every leaf label's classification score (titled with the label names).

As you can see, this method is very similar to the ``test`` method above, just that it calls the ``forward_with_features()`` method we have just implemented instead of the typical ``forward()`` method.

Note how ``all_pooled_features`` only has 24 features as opposed to 768 (which is 768 divided by the pooling kernel size of 32 as specified above).

.. code-block:: python
    :dedent: 0

        def gen_reference_set(self, loader):
            self.eval()
            all_pooled_features = np.empty((0, 24))
            all_targets = np.empty((0), dtype=int)
            all_outputs = np.empty(
                (0, self.classifier.hierarchy.levels[-1]), dtype=float)

            with torch.no_grad():
                for batch_idx, data in enumerate(tqdm(loader)):
                    ids = data['ids'].to(self.device, dtype=torch.long)
                    mask = data['mask'].to(self.device, dtype=torch.long)
                    targets = data['labels']

                    leaf_outputs, pooled_features = self.\
                        forward_with_features(ids, mask)
                    all_pooled_features = np.concatenate(
                        [all_pooled_features, pooled_features.cpu()]
                    )
                    # Only store leaves
                    all_targets = np.concatenate([all_targets, targets[:, -1]])
                    all_outputs = np.concatenate([all_outputs, leaf_outputs.cpu()])

            cols = {
                'targets': all_targets
            }
            leaf_start = self.classifier.hierarchy.level_offsets[-2]
            for col_idx in range(all_pooled_features.shape[1]):
                cols[str(col_idx)] = all_pooled_features[:, col_idx]
            for col_idx in range(all_outputs.shape[1]):
                cols[
                    self.classifier.hierarchy.classes[leaf_start + col_idx]
                ] = all_outputs[:, col_idx]
            return pd.DataFrame(cols)

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
    import torch

    import bentoml
    from bentoml.adapters import JsonInput
    from bentoml.frameworks.transformers import TransformersModelArtifact
    from bentoml.frameworks.pytorch import PytorchModelArtifact
    from bentoml.service.artifacts.common import JSONArtifact
    from bentoml.types import JsonSerializable

    EVIDENTLY_HOST = os.environ.get('EVIDENTLY_HOST', 'localhost')
    EVIDENTLY_PORT = os.environ.get('EVIDENTLY_PORT', 5001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Note the two environment variables here (``EVIDENTLY_HOST`` and ``EVIDENTLY_PORT``). This is to allow the different components of our service to be run both directly on host machine's network as well as being containerised in a Docker network (in which hostnames are not just ``localhost`` anymore). KTT will provide the necessary ``docker-compose`` configuration to set these environment variables to the suitable values, so reading them here and using them correctly is really all we need to do.

Next, we need to implement the service class. It will be a subclass of ``bentoml.BentoService``. All of its dependencies, data (called 'artifacts') and configuration are defined via @decorators, as BentoML internally uses a dependency injection framework.

.. code-block:: python

    # Tell the BentoML exporter what needs to be installed. These will go into
    # the Dockerfile and requirements.txt in the service's folder.
    @bentoml.env(
        requirements_txt_file='models/testmodel/bentoml/requirements.txt',
        docker_base_image='bentoml/model-server:0.13.1-py36-gpu'
    )
    # What this service needs to run: an encoder (DistilBERT), a classifier
    # (our testmodel), the hierarchical metadata and a config variable
    # specifying whether a monitoring server has been exported along.
    @bentoml.artifacts([
        TransformersModelArtifact('encoder'),
        PytorchModelArtifact('classifier'),
        JSONArtifact('hierarchy'),
        JSONArtifact('config'),
    ])
    # The actual class
    class TestModel(bentoml.BentoService):
        """Real-time inference service for the test model."""

        _initialised = False

        # We could also put these in the predict() method, but that will put
        # unnecessary load on the interpreter and reduce our throughput.
        # However, we cannot put them in __init__() as this class will also
        # be constructed without any of the artifacts injected once (in the
        # export() method of the model implementation).
        def init_fields(self):
            """Initialise the necessary fields. This is not a constructor."""
            self.tokeniser = self.artifacts.encoder.get('tokenizer')
            self.encoder = self.artifacts.encoder.get('model')
            self.classifier = self.artifacts.classifier
            # Load hierarchical metadata
            hierarchy = self.artifacts.hierarchy
            self.level_sizes = hierarchy['level_sizes']
            self.level_offsets = hierarchy['level_offsets']
            self.classes = hierarchy['classes']
            # Load service configuration JSON
            self.monitoring_enabled = self.artifacts.config['monitoring_enabled']
            # We use PyTorch-based Transformers
            self.encoder.to(device)
            # Identical pool layer as in the test script.
            self.pool = torch.nn.AvgPool1d(REFERENCE_SET_FEATURE_POOL)

            self._initialised = True

Lastly, we implement the actual predict() API handler as a method in that class, wrapped by a ``@bentoml.api`` decorator that defines the input type (for informing the outer BentoML web server) and microbatching specification.

.. code-block:: python
    :dedent: 0

        # It is HIGHLY recommended that you implement a microbatching-capable
        # predict() method like the one below. Microbatching leverages the GPU's
        # parallelism effectively even in a live inference environment, leading to
        # a ~50x speedup or more.
        # If microbatching is used, the input to this method will be a list of
        # JsonSerializable instead of a single JsonSerializable directly. Simply
        # treat each of them like a row in your test set.
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
            texts = [j['text'] for j in parsed_json_list]
            # Pre-processing: tokenisation
            tokenised = self.tokeniser(
                texts,
                None,
                add_special_tokens=True,  # CLS, SEP
                max_length=64,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
                # DistilBERT tokenisers return attention masks by default
            )
            # Encode using DistilBERT
            encoder_cls = self.encoder(
                tokenised['input_ids'].to(device),
                tokenised['attention_mask'].to(device)
            )[0][:, 0, :]
            encoder_cls_pooled = self.pool(encoder_cls)
            # Classify using our classifier head
            scores = self.classifier(encoder_cls).cpu().detach().numpy()
            # Segmented argmax, as usual
            pred_codes = np.array([
                np.argmax(
                    scores[
                        :,
                        self.level_offsets[level]:
                        self.level_offsets[level + 1]
                    ],
                    axis=1
                ) + self.level_offsets[level]
                for level in range(len(self.level_sizes))
            ], dtype=int)

            predicted_names = np.array([
                [self.classes[level] for level in row]
                for row in pred_codes.swapaxes(1, 0)
            ])

There's one more thing in this method to implement: some code to send the newly-received data-in-the-wild plus our model's scores for it to the monitoring service.
For more information regarding the format of the data to be sent to the monitoring service, please see :ref:`service-spec`.

.. code-block:: python
    :dedent: 0

            if self.monitoring_enabled:
                """
                Create a 2D list contains the following content:
                [:, 0]: leaf target names (left as zeroes)
                [:, 1:25]: pooled features,
                [:, 25:]: leaf classification scores.
                The first axis is the microbatch axis.
                """
                new_rows = np.zeros(
                    (len(texts), 1 + POOLED_FEATURE_SIZE + self.level_sizes[-1]),
                    dtype=np.float64
                )
                new_rows[
                    :,
                    1:POOLED_FEATURE_SIZE+1
                ] = encoder_cls_pooled.cpu().detach().numpy()
                new_rows[
                    :,
                    POOLED_FEATURE_SIZE+1:
                ] = scores[:, self.level_offsets[-2]:]
                requests.post(
                    "http://{}:{}/iterate".format(EVIDENTLY_HOST, EVIDENTLY_PORT),
                    data=json.dumps({'data': new_rows.tolist()}),
                    headers={"content-type": "application/json"},
                )
            return ['\n'.join(row) for row in predicted_names]

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
   torch==1.10.2
   transformers==4.18.0
   numpy==1.19.5

It is always good practice to lock your versions. Only manually update a dependency version when necessary. This prevents breakages, as big Python libraries are known to fight each other over their own dependencies' versions.

For ``dashboard.json``, simply leave it blank for now.

``export_`` methods
^^^^^^^^^^^^^^^^^^^

Time to get back to ``testmodel.py``. We will implement both export schemes: ONNX and BentoML.

Exporting to ONNX is relatively straightforward if not for the fact that transformer models need to be dealt with specially. For this reason, we export the DistilBERT encoder and the classifier head as separate ONNX graphs using different facilities.

.. code-block:: python
	:dedent: 0
	
		def export_onnx(self, classifier_path, encoder_path=None):
			# Don't forget to put your model into evaluation mode!
		    self.eval()
		    # By design, some models don't output to encoder_path.
		    # This model however needs it, so we have to check if the path
		    # was passed.
		    if encoder_path is None:
		        raise RuntimeError('This model requires an encoder path')
		    export_trained(
		        self.encoder,
		        encoder_path
		    )
		    x = torch.randn(1, 768, requires_grad=True).to(self.device)
		    # Export into transformers model .bin format
		    torch.onnx.export(
		        self.classifier,
		        x,
		        classifier_path + 'classifier.onnx',
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
		    # For convenience
		    self.classifier.hierarchy.to_json("{}/hierarchy.json".format(classifier_path))

Exporting as a BentoService is a bit more involved. We will implement it to support an optional monitoring extension powered by the Evidently library. This will be run as a standalone server accepting new data from production to compare with the above reference dataset to compute feature and target drift. To ease this process, KTT has already implemented said standalone server to be customisable (meaning new models can simply write a configuration file to tailor it to their needs and capabilities), as well as automating the file and folder logic for you. All you need to do is to produce two specific pieces of data: a configuration dictionary that lists out the features and classes this model has been trained on, and a fully packed BentoService instance.

We will now use the above facilities to export our new model as a self-contained, standalone classification service.

.. code-block:: python
    :dedent: 0

		def export_bento_resources(self, svc_config={}):
		    self.eval()
		    # Sample input
		    x = torch.randn(1, 768, requires_grad=True).to(self.device)
		    # Config for monitoring service
		    config = {
		        'prediction': self.classifier.hierarchy.classes[
		            self.classifier.hierarchy.level_offsets[-2]:
		            self.classifier.hierarchy.level_offsets[-1]
		        ]
		    }
		    svc = svc_lts.DB_Linear()
		    # Pack tokeniser along with encoder
		    encoder = {
		        'tokenizer': get_tokenizer(),
		        'model': self.encoder
		    }
		    svc.pack('encoder', encoder)
		    svc.pack('classifier', torch.jit.trace(self.classifier, x))
		    svc.pack('hierarchy', self.classifier.hierarchy.to_dict())
		    svc.pack('config', svc_config)
		    return config, svc

Registering, testing & conclusion
---------------------------------

With every part of your model implemented, now is the time to add it to the model list and implement some runner code to get the training and exporting script to use it smoothly. For this, you can refer to :ref:`model-register`.

Be sure to test out every option for your model before deploying to a production environment. Testing instructions can be found at :ref:`test-run`. Afterwards, design a Grafana dashboard and add it to the provisioning system to have your service automatically initialise Grafana right from the get-go.

After this, your model is pretty much complete. If you did it correctly, it should be an integral and uniform part of your own KTT fork and can be used just like any existing (bundled) model.
















