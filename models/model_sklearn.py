"""This file defines functions specific to sklearn-based models."""
import os
from abc import ABC

import logging
from sklearn import metrics
from sklearn_hierarchical_classification.constants import ROOT
import pandas as pd
import json

from models.model import Model


def make_hierarchy(hierarchy_dict):
    """Construct a special hierarchy data structure.

    This structure is for sklearn_hierarchical_classification only.

    Parameters
    ----------
    hierarchy_dict: dict
        The hierarchy dictionary as read from the JSON metadata created
        by data adaptres.

    Returns
    -------
    hierarchy: dict
        A special hierarchy dictionary for sklearn_hierarchical_classification.
    """
    # classes = hierarchy_dict['classes']
    levels = hierarchy_dict['level_sizes']
    offsets = hierarchy_dict['level_offsets']
    # Init level 0 to root list
    hierarchy = {
        ROOT: [
            str(i)
            for i in range(levels[0])
        ]
    }

    for i, level in enumerate(hierarchy_dict['parent_of'][1:]):
        depth = i+1
        for child_idx, parent_idx in enumerate(level):
            # Convert to global space
            gl_child_idx = str(child_idx + offsets[depth])
            gl_parent_idx = str(parent_idx + offsets[depth-1])
            try:
                hierarchy[gl_parent_idx].append(gl_child_idx)
            except KeyError:
                hierarchy[gl_parent_idx] = [gl_child_idx]
    return hierarchy


def get_loaders(
        name,
        config,
        preprocessor,
        shuffle=False,
        verbose=False
):
    """
    Generate 'loaders' for scikit-learn models.

    Scikit-learn models simply read directly from lists. There is no
    special DataLoader object like for PyTorch.

    Parameters
    ----------
    name: str
        Name of the intermediate dataset, for path construction.
    config: dict
        Unused by Sklearn models, but kept for API compatibility.
    preprocessor: None
        Unused by Sklearn models. but kept for API compatibility.
    shuffle: bool
        Unused, as Sklearn "DataLoaders" are just constant DataFrames.
    verbose: bool
        If true, print more detailed information about the loading process.

    Returns
    -------
    train_loader: (pandas.Series, pandas.Series)
        A tuple of inputs and label series, respectively.
    val_loader: None
        Currently, we do not support validation sets for sklearn.
    test_loader: (pandas.Series, pandas.Series)
        A tuple of inputs and label series, respectively.
    hierarchy: dict
        A special hierarchy dictionary for sklearn_hierarchical_classification.
    """
    train_path = 'datasets/{}/train.parquet'.format(name)
    test_path = 'datasets/{}/test.parquet'.format(name)

    targets = []
    if not os.path.exists(train_path):
        if not os.path.exists(train_path + '.dvc'):
            raise OSError('Training set not present and cannot be retrieved.')
        targets.append(train_path + '.dvc')

    if not os.path.exists(test_path):
        if not os.path.exists(train_path + '.dvc'):
            raise OSError('Test set not present and cannot be retrieved.')
        targets.append(test_path + '.dvc')

    if len(targets) > 0:
        os.system('dvc checkout {} {}'.format(
            ' '.join(targets), '-v' if verbose else ''))

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    # Generate hierarchy
    with open(
            'datasets/{}/hierarchy.json'.format(name), 'r'
    ) as hierarchy_file:
        hierarchy = make_hierarchy(json.load(hierarchy_file))

    X_train = train['name'].apply(lambda n: preprocessor(n)['text'])
    X_test = test['name'].apply(lambda n: preprocessor(n)['text'])

    y_train = train['codes'].apply(
        lambda row: row[-1]
    )
    y_test = test['codes'].apply(
        lambda row: row[-1]
    )

    return (X_train, y_train), None, (X_test, y_test), hierarchy


def get_metrics(test_output, display='log', compute_auprc=True):
    """Compute leaf-level metrics for Scikit-learn models.

    The following metrics are computed:

    - Leaf accuracy (accuracy at the leaf level)
    - Leaf precision (precision at the leaf level)
    - (optionally) AU(PRC) (at the leaf level)

    Parameters
    ----------
    test_output: dict
        A dict containing the following keys:

        - ``predictions``: numpy.ndarray of size (len(test_set), 1)
            Names of labels classified by the model.
        - ``targets``: numpy.ndarray of size (len(test_set), 1)
            Names of ground-truth labels.
        - ``scores``: Numerical scores for each label, which can be acquired
            using sklearn models' predict_proba().
        - ``targets_b``: torch.LongTensor of shape (minibatch, len(hierarchy.classes))
            List of binarised target vectors in global space.

    display: string
        Optional display mode, given as string. There are three options:

        - ``log``: Write metrics to the default log output.
        - ``print``: Print metrics to the screen.
        - ``both``: Do both of the above.

    compute_auprc: bool
        Whether to compute the AU(PRC) metric at the leaf level. If true, the
        returned array has an additional metric at the end, making it 5 elements
        long.

    Returns
    -------
    metrics: np.ndarray of shape (4) or (5)
        The list of metrics computed in the order listed above. Note that since
        we do not compute path-average metrics for sklearn models, the third and
        fourth items are None.

    """
    leaf_accuracy = metrics.accuracy_score(
        test_output['targets'],
        test_output['predictions']
    )
    leaf_precision = metrics.precision_score(
        test_output['targets'],
        test_output['predictions'],
        average='weighted',
        zero_division=0
    )
    leaf_auprc = metrics.average_precision_score(
        test_output['targets_b'],
        test_output['scores'],
        average="micro"
    )
    if display == 'print' or display == 'both':
        print("Leaf accuracy: {}".format(leaf_accuracy))
        print("Leaf precision: {}".format(leaf_precision))
        print("Leaf AU(PRC): {}".format(leaf_auprc))
    if display == 'log' or display == 'both':
        logging.info("Leaf accuracy: {}".format(leaf_accuracy))
        logging.info("Leaf precision: {}".format(leaf_precision))
        logging.info("Leaf AU(PRC): {}".format(leaf_auprc))

    return (leaf_accuracy, leaf_precision, None, None, leaf_auprc)


class SklearnModel(Model, ABC):
    """Convenience class wrapping the Model abstract class.

    It implements two class methods that are the same for all Sklearn models.
    """

    @classmethod
    def get_dataloader_func(cls):
        """Return KTT's PyTorch-compatible ``get_loaders`` implementation."""
        return get_loaders

    @classmethod
    def get_metrics_func(cls):
        """Return KTT's PyTorch-compatible ``get_metrics`` implementation."""
        return get_metrics

    def to(self, device):
        """All sklearn models are CPU-only, so this method is unused.

        This class implements it as a passthrough method so you don't have to.
        """
        return self
