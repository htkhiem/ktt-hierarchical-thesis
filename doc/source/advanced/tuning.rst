.. Ray Tune guide

Automatic hyperparameter tuning
===============================

KTT is able to automatically find the best hyperparameters for a model with a given dataset, within a set of sane default ranges empirically determined by model implementers. To achieve this, KTT enlists the help of Ray's Tune library. The best set of hyperparameters are then printed on screen and logged to a file. Users can also additionally allow the tuning component to overwrite the default hyperparameter configuration with the best found, such that normal training will immediately take advantage of these hyperparameters.

CLI usage
---------

Please refer to :ref:`tune-cli`.

Tune configuration format
-------------------------

KTT's configuration format for the tuning script is different from the one used for normal training. Each model still gets a JSON object with the model identifier as the key, but within it, there are three parts:

- The global configuration part: these contain data-related configuration, just like the normal training configuration schema. These are not tunable.
- The ``range`` object, which contains all the hyperparameters but in terms of "sane" ranges as defined by the implementers of the model. Hyperparameter search will be performed within these ranges. Three kinds of ranges are supported:

    - Fixed value (any data type): these hyperparameters are not tunable.
    - List of values (any data type, any length): a set of possible values. Hyperparameters specified with this kind of range will have their value sampled at random.
    - List of two numbers: if your range is in this form, then it can be used to define the lower and upper bound of a continuous range.
- The ``mode`` object, which defines the type of sampling to use for each hyperparameter range as configured in the above ``range`` object. Three modes are currently supported, listed in the same order as the ranges they support above:
    - ``fixed``: Not tuneable - keep constant.
    - ``choice``: Pick one value out of those specified.
    - ``uniform``: Sample a value from the continuous space bounded by the two numbers given in the list.

An example configuration:

.. code-block:: json

    "db_achmcnn": {
        "display_name": "DistilBERT + Adapted C-HMCNN",
        "train_minibatch_size": 16,
        "val_test_minibatch_size": 64,
        "max_len": 64,
        "range": {
            "encoder_lr": [0.00001, 0.00006],
            "classifier_lr": [0.0002, 0.001],
            "h_dropout": [0.0, 0.75],
            "h_nonlinear": "relu",
            "h_hidden_dim": 512,
            "h_layer_count": [1, 2, 3, 4]
        },
        "mode": {
            "encoder_lr": "uniform",
            "classifier_lr": "uniform",
            "h_dropout": "uniform",
            "h_nonlinear": "fixed",
            "h_hidden_dim": "fixed",
            "h_layer_count": "choice"
        }
    },
