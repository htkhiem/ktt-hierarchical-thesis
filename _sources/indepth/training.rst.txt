.. Training instructions page.

Training stage
==============

This is where most of the computations happen - the model training stage. KTT comes bundled with seven models in total and various options to train them.

The process
-----------

Starting from an intermediate (pre-processed) dataset, the training process can be summarised as follows:

1. Load the individual training, validation and test subsets into memory.
2. Generate suitable loader objects.
	
	- For PyTorch, KTT generates one ``torch.utils.dataset.DataLoader`` object for each of the above subsets. These dataloaders in turn take a :py:class:`models.model_pytorch.PyTorchDataset` object which contains a reference to the subset (loaded into memory as a Pandas dataframe). The :py:class:`models.model_pytorch.PyTorchDataset` handles model-specific preprocessing of each row, and the ``DataLoader`` object randomly samples and marshalls such processed rows into minibatches.
	- For Scikit-learn, KTT simply passes the Pandas dataframes as-is.
3. Generate hierarchy from saved hierarchical metadata. KTT internally uses the :py:class:`utils.hierarchy.PerLevelHierarchy` class to aid in storing, importing and exporting this metadata for use in training and checkpointing.
4. Create a model instance from the above hierarchy.
5. Train the model using its own ``fit()`` method.
	- Bundled PyTorch models (as with any deep-learning models) are trained through multiple epochs. For each epoch, the model is first trained on the training set, then validated using the validation set. The validation set's metrics are logged automatically. After having trained for the configured number of epochs, the model is then ran over the test set.
	- Scikit-learn models simply use their ``fit()`` method.
6. If a reference set is requested at the CLI, then the model is again run over the test set, but this time outputting additional information for generating the reference set.
	
Classes
-------

Common classes
^^^^^^^^^^^^^^

.. autoclass:: utils.hierarchy.PerLevelHierarchy
	:members:

.. _pytorch-utils:

PyTorch utilities
^^^^^^^^^^^^^^^^^
.. automodule:: models.model_pytorch
	:members:
	:inherited-members:
	
.. _sklearn-utils:

Scikit-learn utilities
^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: models.model_sklearn
	:members:
	:inherited-members:
