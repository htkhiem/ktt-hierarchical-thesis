.. Model exporting page.

Exporting your models
=====================

KTT is not an inference system - it produces models and additionally generates inference systems based around those models for you. KTT supports exporting trained models in two ways:

- Open Neural Network Exchange (ONNX) format - an open, vendor-neutral format that can be deployed to almost everywhere.
- BentoML - which internally also partly uses ONNX to store models. This is to be used with our bundled REST API-powered inference system.

ONNX exporting
--------------

This is the way to go if you intend to use KTT-generated models with an existing inference service, such as Azure ML, or if you already have your own framework set up and just need the models themselves.
All bundled models in KTT except one (:doc:`/models/tfidf_hsgd`) supports exporting to ONNX. The exact file structure is not enforced to allow model implementers more freedom in how they export their designs, but bundled PyTorch models generally follow a two-graph scheme:

- The encoder (currently DistilBERT is bundled) is exported to an ONNX graph in ``outputs/modelname_datasetname/encoder`` in several different files based on their own exporting logic.
- The classifier head (any of the five PyTorch models) is exported to a single file, ``outputs/modelname_datasetname/classifier/classifier.onnx``.

Currently, KTT exports to ONNX using opset version 11. All ONNX models support dynamic minibatch axes in that they can be passed one or many data items at once. This is especially useful should you want to use the exported models for minibatched offline inference, or dynamically microbatched real-time inference.

BentoML exporting
-----------------

KTT is able to construct BentoServices from its models based on resources defined by the models themselves. A BentoService can be constructed either with or without a production-time performance monitoring system. When built with such system integrated, a deployment would look something like this out-of-the-box:

.. image:: bentoml-deploy.svg
   :width: 800
   :alt: A full deployment of a BentoService on a single host.
   
A full BentoService deployment consists of four smaller services linked together in a Docker network: ``inference``, ``monitoring``, ``prometheus`` and ``grafana``. The main workings of the deployed service can be described as follows:

1. User sends a JSON request to the inference service. It produces an answer, respond to the user, and log both the user input (in the form of feature vectors - not raw text) and its own answers (in the form of label scores) to the monitoring service. A background process of the inference service also keeps track of response time and other system performance metrics.
2. The monitoring service accumulates data received from the inference service into a rolling dataset of N newest rows. This dataset is periodically compared against a *reference dataset* produced during training of the model this service uses, and the results are stored temporarily locally.
3. The Prometheus service, which as its name implies is a Prometheus time-series database instance, periodically fetches metrics from both the inference and monitoring services.
4. The Grafana dashboard periodically fetches the latest data stored within Prometheus, transforms them, and visualises them as a user-friendly performance monitoring dashboard.

By default, all four services communicate to each other via a Docker network, and all are directly accessible from the host machine via the same port number. With a bit of modification to the ``docker-compose.yaml`` preset produced by KTT, one can also host these four services separately if desired.

Currently, BentoML services do not make use of ONNX graphs due to BentoML LTS's rather buggy nature with the ONNX + CUDA runtime combination. As such, they directly use serialised versions of the models as implemented by BentoML.

Packaging
~~~~~~~~~

KTT-produced BentoServices are easily packaged - they are self-contained within their own directory in the ``build`` folder. Sinply zip them up (or use any archival format you like) and you have a portable system.

Advanced users might want to produce their own Docker Compose configurations, or use Kubernetes instead.
