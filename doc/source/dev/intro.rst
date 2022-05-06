.. Dev section - introduction.

Developing new models
===================================================

We have designed the system to be easily extensible - new models can be added with relative ease. 

Frameworks
----------

Bundled models are implemented using either PyTorch (for GPU-based models) or Scikit-learn (everything else). New models can be easily implemented using either of these frameworks. Implementing using other frameworks (such as Tensorflow) should be possible (as the common core of KTT is not hardwired to any particular framework) but has not been tested. Furthermore, you will need to add and keep track of your own dependencies yourself.

Should you decide to use another framework, please refer to the section at the end of this page.


.. _model-struct:

General model folder structure
------------------------------

Models should have a fixed and uniform folder structure regardless of framework. This is not mandatory, but is highly encouraged to enable easier maintenance and improve code consistency. Bundled models follow the following folder structure:


.. code-block:: bash

    models
    └── model_name
        ├── __init__.py
        ├── bentoml
        │   ├── __init__.py
        │   ├── dashboard.json
        │   ├── evidently.yaml
        │   └── svc_lts.py
        └── model_name.py	
        
As one can see, each model is put in their own folder, whose name is also the model's identifier to be used in the common training and exporting controller. All model folders are put inside the ``./models`` folder.

An explanation of what each file and folder is:

- ``__init__.py`` files are there to ensure Python reads each folder as a module, allowing the model to be imported.
- ``model_name.py`` is where the model is implemented as a ``Model`` object. This is also where most of the work happens. Note that it has the same name as the model folder.
- ``bentoml/`` contains all BentoML-related code and resources. These files and resources are needed to build a BentoService from this model.
- ``bentoml/dashboard.json`` is the Grafana dashboard's JSON schema. This file will be automatically *provisioned* for the Grafana instance when we start the Dockerisation process.
- ``evidently.yaml`` is a *template* configuration file for the Evidently monitoring service. By itself, it contains dataset-agnostic configuration parameters tailored to this model. When the service is built, a copy is made of this file and additional dataset-dependent configuration added. That copy will reside in the built BentoService.
- ``svc_lts.py`` is the monitoring application. It receives new data (feature inputs and resulting scores) from the main inference service, compares against a *reference dataset* and computes data drift/target drift metrics to be sent to the Prometheus database instance. It is a separate web service from the main inference service. 

With that out of the way, we will now deep-dive into how to implement each part of a model:

.. _model-class:

The model itself
----------------

Every model must subclass a framework-specific abstract model class, all of which in turn subclass ``models.model.Model``. This serves as the baseline requirement, or the minimum set of features a model must implement so they could be used with KTT's training and exporting facilities.

.. autoclass:: models.model.Model
   :members:
   :special-members:

Unless you are implementing your model in a framework other than PyTorch or Scikit-learn, you need not bother with ``get_dataloader_func`` and ``get_metrics_func``. The rest will be detailed below.

Checkpointing
-------------

This part gives detail on how to implement ``save``, ``load`` and ``Model.from_checkpoint``.

If either ``path`` or ``best_path`` is specified for the ``fit`` method, a model's training session must produce at least one checkpoint. Checkpoints must contain enough data to fully replicate the model and its state, including the state of any training process up to that moment in time (for example, Adam optimiser state dictionaries). For example, a PyTorch model checkpoint must contain not just ``state_dict`` fields but also sufficient metadata regarding the topology (number of layers, layer sizes and so on).

Model checkpoints (or pickled trained models) for a particular dataset must be saved to ``./weights/<model_name>/<dataset name>``. For example, a model named ``abc`` trained against an intermediate dataset named ``data001`` must save its weights in ``./weights/abc/data001``. These are to be implemented at the ``fit`` method, which will pass suitable paths to the ``save`` method in the same instance to produce checkpoints at those paths.

Checkpoint file names must abide by the following convention:

    - Best-performing checkpoints (by your own metric, for example the smallest validation loss value): ``best_YYYY-MM-DDTHH:MM:SS.<extension>``.
    - The last checkpoint (produced at the last epoch): ``last_YYYY-MM-DDTHH:MM:SS.<extension>``.

In other words, the checkpoint name is ``best | last`` plus an ISO8601 datetime string (truncated to seconds) and then the file extension. The exact extension depends on the format you choose to save your model's checkpoint in. All bundled models use Pickle to serialise their models and as such use the ``.pt`` extension.

.. note::
   Models that do not generate in-progress checkpoints (such as Scikit-learn models whose training process is a simple blocking ``fit()`` call) can produce their only checkpoint labelled as either ``best`` or ``last``. However, since the export script defaults to looking for ``best`` checkpoints, it would be more convenient to use ``best``. This would allow you to call the export script for these models without having to specify an additional option at the export script.

Preprocessing needs
-------------------
Should your model require the dataset to be preprocessed in any way (for example, tokenisation for DistilBERT and stemming for simple term-based models), implement such logic by subclassing the ``BasePreprocessor`` class in ``utils/encoders/encoder.py``:

.. autoclass:: utils.encoders.encoder.BasePreprocessor
   :members:
   :special-members:
   
With your logic implemented, instruct your model to use it by implementing the ``get_preprocessor`` method. This method returns an instance of your preprocessor class and will be called on every new dataset loading. That instance will be used to preprocess that dataset before its data is fed to your model.

If your model directly takes in raw text from the datasets, simply skip this method. Its default implementation in the ``Model`` class simply returns an instance of the above ``BasePreprocessor`` which does nothing aside from putting your input text in a dictionary at key ``text``.

Exporting
---------

This part gives detail on how to implement the ``export`` method and the BentoService.

KTT models should be able to export themselves into 2 formats:

    - ONNX: Open Neural Network eXchange, suitable for deploying to existing ML services with an ONNX runtime. A model may be exported into one or more ``.onnx`` files.
    - BentoML: Not strictly a 'format per se, but rather a packaging of the model with the necessary resources to create a standalone REST API server based on the BentoML framework.

Your model should support both formats, but at the minimum, it should support one of them (because what good is a model that can only be trained but not used?).

ONNX
~~~~

Exporting to ONNX should be a simple process as it does not involve the rest of the system that much. Simply use the ONNX converter tool appropriate for your framework (for example, ``skl2onnx`` for ``sklearn`` and ``torch.onnx`` for PyTorch).

By default, exported ONNX models should be saved at ``./outputs/<model name>_<dataset_name>/``. For example, a model named ``abc`` trained against a dataset named ``data001`` should export its ``.onnx`` graphs to ``./outputs/abc_data001``.

BentoML
~~~~~~~

Exporting into a BentoML service is more involved, and also gives you more decisions to make. There are two main approaches regarding the format that you can export your model core into for your BentoService implementation to later use.

    - Reuse the ONNX graphs. This is possible, but you might run into problems with BentoML's internal ONNX handling code not playing well with CUDA and TensorRT (at least in the current version). For non-GPU models, this is generally stable and could save you a bit of time (due to slightly better optimisation from an ONNXRuntime) and storage space (depending on your checkpoint format).
    - Directly serialise your model and implement your BentoService's runner function to behave just like your ``test`` method. Depending on framework and code structure, you might still achieve equal performance as the ONNX approach. This has the added bonus of allowing you to adapt the code from your ``test`` method into the BentoService, saving implementation time. It is also less fussy and is more likely to work well for GPU-enabled models.

.. tip::
   KTT internally uses BentoML 0.13.1 (the LTS branch). You can find specific instructions for your framework in `their documentation <https://docs.bentoml.org/en/0.13-lts/frameworks.html>`_.

The implementation for BentoML exporting is split into three parts: the ``svc_lts.py`` source file, the reference dataset's generation plus configuration files (optional), and the ``export`` method. The expected result after running the export script with this model would be a new BentoService in the ``./build`` folder with the following structure:

.. code-block::

   build
   └── <model name>_<dataset name>
        ├── docker-compose.yaml
        ├── grafana
        │   └── provisioning
        │       ├── dashboards
        │       │   ├── dashboard.json
        │       │   └── dashboard.yml
        │       └── datasources
        │           └── datasource.yml
        ├── inference
        │   ├── bentoml-init.sh
        │   ├── bentoml.yml
        │   ├── (...)
        │   ├── Dockerfile
        ├── monitoring
        │   ├── Dockerfile
        │   ├── evidently.yaml
        │   ├── monitoring.py
        │   ├── references.parquet
        │   └── requirements.txt
        └── prometheus
            └── prometheus.yaml

or this, if it was built without support for monitoring:

.. code-block::

   build
   └── <model name>_<dataset name>
        └── inference
            ├── bentoml-init.sh
            ├── bentoml.yml
            ├── (...)
            └── Dockerfile

To keep things nice and tidy, we recommend that you create a subfolder within your model folder to store BentoML-specific files just like the example general folder structure at :ref:`model-struct`.

.. _service-spec:

The service implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``svc_lts.py`` file contains the definition of the BentoService for this model. The following is a rough description of what you need to implement in this file:

    1. Define a subclass of ``bentoml.BentoService`` preferably named just like your model class (as it will be the name used by the exported BentoService's internal files and folders).
    2. Based on what your model requires, define Artifacts for this class via the ``@bentoml.artifacts`` decorator. Artifacts are data objects needed for the service, such as the serialised model itself, a metadata JSON file, or some form of configuration.
    3. Define a ``predict`` method that accepts one of BentoML's ``InputAdapters`` (see `here <https://docs.bentoml.org/en/0.13-lts/api/adapters.html>`_) and returns a single string, preferably a newline-separated list of class names in hierarchical order (example: ``Food\\nPasta\\nSpaghetti``). This method will contain the code needed to preprocess and feed the data received passed as an ``InputAdapter`` by the outer BentoServer into the model, then get the results out of the model and postprocess into said string format. You can access the Artifacts from within this method (or any method within this class) by retrieving ``self.artifacts.<artifact name>``. Remember to wrap this method in a ``@bentoml.api`` decorator.
    4. (optional) If you decide to implement monitoring capabilities for your model's BentoService, make the ``predict`` method send new data to the monitoring app on every request processed. The data to be sent is a JSON object containing a single 2D list at field ``data``. This 2D list has shape ``(microbatch, reference_cols)`` where ``microbatch`` is the number of microbatched rows in this request (1 if you do not enable microbatching), and ``reference_cols`` is the number of columns in your *reference dataset* (more on this later). As you might have guessed, this 2D array is basically a tabular view of a new section to be appended to a *current dataset* which will be compared against said *reference dataset*. The monitoring service's hostname and port are exposed via the ``EVIDENTLY_HOST`` and ``EVIDENTLY_PORT`` environment variables. If these variables have not been set, default to ``localhost:5001``.

.. note::
   We chose to pass them via environment variables so as to give the BentoService more flexibility. If you run everything right on the host machine, the Evidently monitoring app can be reached at ``localhost:5001``. However, when Dockerised and run in a ``docker-compose`` network, each container is on a different host instead of ``localhost``. By making the BentoService read environment variables, our ``docker-compose.yaml`` file can pass the suitable hostname and port to it, allowing it to continue functioning normally in both cases.

With the service implemented, we can move on to implementing logic for automatically generating the *reference dataset*. A reference dataset is any dataset that is representative of the data the model instance was trained on. Bundled models simply use the test subset as the reference dataset. However, due to requirements from Evidently (the model metrics framework used by KTT), the reference dataset instead needs to contain numerical features (for the ``feature_drift`` report), with each feature being given one column. In addition to that, it may also require the raw classification scores (the numerical results the model outputs) if you choose to specify ``categorical_target_drift`` reports as part of the metrics to compute and track - again, each class gets its own scores column.

.. _bentoml-config:

The service configuration files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you chose not to include monitoring capabilities with this model, you may safely skip this part. If you do want to have them, however, then there are two configuration files to add to your model folder: ``evidently.yaml``, and ``dashboard.json``.

The ``evidently.yaml`` configuration template follows Evidently's service configuration format. Technically, you can also use JSON or any other format, as you will be the one implementing the parsing code later. However, since the Evidently monitoring app itself uses YAML, it is best to just stick to that. There are a number of parameters that **must** be provided by the template. A sample templated configuration for DB-BHCN, which contains all of them, can be seen below:

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

A simple explanation of these fields:

    - ``reference_path`` is a relative path to the reference dataset, whose generation we will implement shortly. The dataset must be copied along with the monitoring app to the built service's folder. This path is relative to the monitoring app itself, which is the ``monitoring.py`` file within ``./build/<service name>/monitoring``. It is best to simply copy the reference dataset to the same folder as the monitoring app.
    - ``min_reference_size`` specifies how many rows must be collected by the metrics app (from user inputs and scores forwarded by the inference service) for the metrics to start being computed. It should be at least ``window_size`` or greater.
    - ``moving_reference`` here is set to false, as we need a fixed reference set to analyse data drift.
    - ``window_size`` specifies how many collected rows to use for comparison with the reference set. Here, the latest 30 rows are used. Smaller values reduce RAM usage and metrics computation time, while larger values may help capture wider shifts in trends.
    - ``calculation_period_sec`` is in seconds and specifies how frequently metrics should be computed based on the latest ``window_size`` collected rows. We recommend setting it to a high value if you do not expect your production environment to change quickly, as this has a significant performance impact.
    - ``monitors`` is a list of Evidently reports to compute. Each report in this context is a set of metrics, which can then be displayed in a Grafana dashboard. The following are available from our Evidently monitoring app:
        .. code-block:: python

            monitor_mapping = {
                "data_drift": DataDriftMonitor,
                "cat_target_drift": CatTargetDriftMonitor,
                "regression_performance": RegressionPerformanceMonitor,
                "classification_performance": ClassificationPerformanceMonitor,
                "prob_classification_performance": ProbClassificationPerformanceMonitor,
            }

    Detailed information for each of these reports can be found `in the official Evidently documentation <https://docs.evidentlyai.com/reports>`_.


The second configuration file is the Grafana dashboard layout (``dashboard.json``). We however do not recommend you create this file by hand. Instead, you can wait until you can boot up your service and log into the Grafana instance so you can create it interactively.
More on that later in :ref:`grafana-design`.

.. _reference-set:

The reference dataset
^^^^^^^^^^^^^^^^^^^^^

To generate the reference dataset, you must implement an additional method, called the ```gen_reference_set(self, loader)`` method, which will be called by the training script after the training script and test script if the user specifies ``--reference`` or ``-r``. This is quite similar to the ``test`` method in that it also runs the model over a dataset (passed as a 'loader' of your choice), but it also records the numerical features (for example, from a Tf-idf vectoriser, or a DistilBERT instance) along with numerical classification scores. For reference, you may want to take a look at DB-BHCN's version:

.. autoclass:: models.DB_BHCN
   :noindex:
   :members: gen_reference_set

The exact reference set schema depends on your choice of Evidently reports and also your model's design. DB-BHCN for example generates a reference dataset containings firstly a ``targets`` column (ground truths, using textual class names), 24 average-pooled feature columns (from the 768 features produced by its DistilBERT encoder) named ``0`` to ``23`` (in string form), and classification score columns, one for each leaf-level class, with the column names being the string names of the classes themselves.

The resulting reference set must be in Parquet format (``.parquet``) named similarly to the **last checkpoint**, with ``_reference`` added. For example, if the last checkpoint is named ``last_2022-04-01T12:34:56.pt``, then the reference dataset must be named ``last_2022-04-01T12:34:56_reference.parquet``.

.. _model-export-general:

The ``export`` method
^^^^^^^^^^^^^^^^^^^^^

In the ``export`` method, if ``bento=True``, you should do the following:

    1. If ``reference_set_path`` is passed, it means the user at the export script has enabled monitoring and there is a reference dataset available (monitoring is not possible without one, so if there isn't one, the export script will simply pass ``None``). Load in the ``evidently.yaml`` template file and write a list of scores column names to the ``prediction`` field. If you chose to follow the default reference dataset schema above, that would be a list of all leaf class names. Then, construct a ``dict`` (we'll call it ``config``) with the following contents:

       - ``reference_set_path``: path to the reference dataset.
       - ``grafana_dashboard_path``: path to the ``dashboard.json`` file. **Note:** since the export script is run from ``./``, specify the relative path from there, not from the model source file.
       - ``evidently_config``: the ``dict`` containing the template config file's contents, with the above ``prediction`` field.

    2. Initialise the service folder structure by calling ``utils.build.init_folder_structure(path, config)``, passing it the path to the built service and the above ``config`` dict object. It is recommended to use ``./build/<model_name>_<dataset_name>`` for the build path. This utility function will also return a path for you to save your BentoService to.

       .. autoclass:: utils.build.init_folder_structure

     If there is no reference dataset or monitoring is not enabled (i.e. ``reference_set_path is None``), simply pass only the path and leave the config parameter blank.

    3. Initialise a BentoService instance (as in importing the above ``svc_lts.py`` file as a module and constructing the BentoService-based class within). Pack all necessary resources into its artifacts.

       - PyTorch modules should be JIT-traced using ``torch.jit.trace`` before packing into a ``PytorchArtifact``.
       - Configuration files or metadata should be packed as strings.

    4. Save the BentoService to the path returned by ``utils.build.init_folder_structure(path)``.

Specifying your hyperparameters (optional)
------------------------------------------

Some models might have tunable hyperparameters. KTT has facilities to automatically retrieve values from a ``./hyperparameters.json`` file. Each model gets a JSON object with their own identifier as the key. You can add your model's hyperparameters to this file, and then tell KTT to load them in at the beginning of the training session to initialise your model.

You can either directly tune your model by modifying this file, or implement automatic hyperparameter tuning in your training script and use this file to supply starting values.

Models without tunable hyperparameters can skip this step.

.. _model-register:

Registering your model with the rest of the system
--------------------------------------------------

Now that you have fully implemented your model, it is time to inform the training and exporting scripts of its existence and also on how to run it.

The model lists
~~~~~~~~~~~~~~~

Edit ``./models/__init__.py`` and add your model to it. There are three places to do so:
    1. First, import your model class (the one subclassing ``Model``). This allows the training and exporting code to shorten the import path to just ``from models import YourModelClass`` instead of ``from models.your_model.your_model import YourModelClass``. Refer to how the bundled models are imported to import your own.
    2. Then, add your model identifier (the model folder name) to the appropriate ``MODEL_LIST``. Currently, there is the ``PYTORCH_MODEL_LIST`` and ``SKLEARN_MODEL_LIST``.

        .. note::

            TODO: Add instructions for implementing a model outside of these frameworks.

    3. Lastly, add your model class name (not the folder name) to ``__all__``. If your model needs to expose special functions that are not available.

The training script
~~~~~~~~~~~~~~~~~~~

Edit ``./train.py`` at the following places:

    - Implement code to load and train your model depending on whether its identifier was in ``model_lst``. Existing code should act as a guideline. A general frame for a PyTorch model would look something like this:

      .. code-block:: python

         if 'model_name' in model_lst:
             # If your model has hyperparameters in ./hyperparameters.json:
             # config = init_config('model_name', 'Pretty display name')
             ModelClass = __import__('models', globals(), locals(), [], 0).ModelClass
             for dataset_name in dataset_lst:
                (
                    train_loader, val_loader, test_loader, hierarchy, config
                ) = init_dataset(
                    dataset_name, model_pytorch.get_loaders, config
                )
                model = ModelClass(hierarchy, config).to(device)
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
    - Optionally modify the Click documentation for the ``-m`` option to add your model's identifier and display name.

The exporting script
~~~~~~~~~~~~~~~~~~~~

Edit ``./export.py`` at the following places:

    - Implement code to load and export your model depending on whether its identifier was in ``model_lst``. Existing code should act as a guideline. A general frame for a PyTorch model would look something like this:

      .. code-block:: python

        if 'model_name' in model_lst:
            click.echo('{}Exporting {}...{}'.format(
                cli.BOLD, 'Pretty display name', cli.PLAIN))
            ModelClass = __import__(
                'models', globals(), locals(), [], 0).ModelClass
            model = ModelClass.from_checkpoint(
                get_path('model_name', dataset_name, best=best, time=time),
            ).to(device)
            if monitoring:
                reference_set_path = get_path(
                    'model_name', dataset_name, time=time, reference_set=True)
                if reference_set_path is not None:
                    model.export(dataset_name, bento, reference_set_path)
                else:
                    model.export(dataset_name, bento)

    - Optionally modify the Click documentation for the ``-m`` option to add your model's identifier and display name.

.. _test-run:

Test-run your model
-------------------
If all goes to plan, you can now call on ``train.py`` and ``test.py`` with your model just like any of the bundled models. Train it on a preprocessed dataset and check if its checkpoints are in the correct format. Ensure that it can load, save and export smoothly.

If you implemented BentoService exporting, you can test-run the built service in two ways:

    - Without monitoring capabilities: either run the inference service directly using ``bentoml serve``, or run as a Docker container using the supplied Dockerfile.

      .. code-block:: bash

        cd ./build/<model_name>_<dataset_name>/inference

         bentoml serve ./
         # or use the production gunicorn server
         bentoml serve-gunicorn ./
         # or as a Docker container
         docker image build .
         docker run -p 5000:5000 <built image ID>

    - With monitoring capabilities: fire up your entire service using the autogenerated ``docker-compose`` script:

      .. code-block:: bash

        cd ./build/<model_name>_<dataset_name>
        docker-compose up

      This will Dockerise the inference app and monitoring app (if not already), download and run Prometheus and Grafana, and configure them all to fit together nicely in a Docker network.

The following ports are exposed by the service:

    - ``5000``: the inference service. ``POST`` inference requests to its ``/predict`` endpoint. The format of the request is whatever you decided on in your BentoService implementation. BentoML's server also returns API- and process-related metrics through the ``/metrics`` endpoint.
    - ``5001``: the monitoring service. It returns data-related metrics through its ``/metrics`` endpoint. Only available in monitoring-enabled services.
    - ``9090``: the Prometheus database control panel. Only available in monitoring-enabled services.
    - ``3000``: the Grafana control panel. Only available in monitoring-enabled services.

.. _grafana-design:

Grafana dashboard design (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you designed your model with monitoring capabilities, now is the time to start designing your Grafana dashboard. Log into Grafana (by default exposed at ``localhost:3000``) with default credentials (``admin`` for both username and password - remember to change them!). A Prometheus data source should already have been included, which connects to the Prometheus instance in your Docker network, which in turn fetches metrics regularly from the inference and monitoring services' ``/metrics`` endpoint.

You can now create your dashboard from this data source, using metrics names returned by the ``/metrics`` endpoints above. You can also import the JSON schema of an existing dashboard from a bundled model to learn how to display them.

Once everything is done and running, export your dashboard as a JSON file (remember to tick that external exporting option). Place the JSON in your model's folder (preferably following the standard folder structure at the beginning of this guide) and rename it if necessary.

Testing automatic dashboard provisioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The expected starting state of a completed BentoService from KTT includes a fully provisioned Grafana instance. This means you should ensure that your Grafana dashboard is loaded in and running without any user intervention right from service startup. To facilitate this, ensure that your ``export`` method correctly passes the path to the dashboard JSON to the ``init_folder_structure`` function. This function would then copy the JSON to ``./build/<model_name>_<dataset_name>/grafana/provisioning/dashboards``.

After finishing designing your dashboard, exporting it, placing the JSON in the correct location and specifying the path in your ``export`` method implementation, you should repeat the whole exporting process. First, remove all previously-built service files. Then, remove all related Docker volumes using ``docker volume prune``. Finally, export your service as usual and ``docker-compose up`` to start it. Log into your Grafana dashboard again (the username and password should have been reverted to the default credentials - if not, you have not fully cleared the previous service's data) and see if the dashboard is already there. If it is, congratulations! You now have a fully working model and BentoService exporting process!

Framework-specific guides
-------------------------

The above instructions only cover parts that are common between all frameworks. See below for in-depth guides for each framework:

.. toctree::
    pytorch/add_model
    sklearn/add_model
