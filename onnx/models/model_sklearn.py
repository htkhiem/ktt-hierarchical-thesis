"""This file defines special functions for sklearn-based models."""
import logging
from sklearn import metrics
from sklearn_hierarchical_classification.constants import ROOT
import pandas as pd
import json


def make_hierarchy(hierarchy_dict):
    """Construct a special hierarchy data structure.

    This structure is for sklearn_hierarchical_classification only.
    """
    classes = hierarchy_dict['classes']
    levels = hierarchy_dict['level_sizes']
    offsets = hierarchy_dict['level_offsets']
    # Init level 0 to root list
    hierarchy = {
        ROOT: [
            classes[i]
            for i in range(levels[0])
        ]
    }

    for i, level in enumerate(hierarchy_dict['parent_of'][1:]):
        depth = i+1
        for child_idx, parent_idx in enumerate(level):
            # Convert to global space
            gl_child_idx = child_idx + offsets[depth]
            gl_parent_idx = parent_idx + offsets[depth-1]
            try:
                hierarchy[classes[gl_parent_idx]].append(classes[gl_child_idx])
            except KeyError:
                hierarchy[classes[gl_parent_idx]] = [classes[gl_child_idx]]
    return hierarchy


def get_loaders(
        dataset_name,
        config,
        verbose=False
):
    """
    Generate 'loaders' for scikit-learn models.

    Scikit-learn models simply read directly from lists. There is no
    special DataLoader object like for PyTorch.
    """
    train = pd.read_parquet('../datasets/{}/train.parquet'.format(dataset_name))
    test = pd.read_parquet('../datasets/{}/test.parquet'.format(dataset_name))
    # Generate hierarchy
    with open('../datasets/{}/hierarchy.json'.format(dataset_name), 'r') as hierarchy_file:
        hierarchy = make_hierarchy(json.load(hierarchy_file))

    X_train = train['name']
    X_test = test['name']
    y_train = train['codes'].apply(
        lambda row: row[-1]
    )
    y_test = test['codes'].apply(
        lambda row: row[-1]
    )

    return (X_train, y_train), (X_test, y_test), hierarchy


def get_metrics(test_output, display='log', compute_auprc=True):
    """Specialised metrics function for scikit-learn model."""
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
