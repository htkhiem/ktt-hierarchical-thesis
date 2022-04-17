import os

import torch
import pandas as pd
from sklearn import metrics
import numpy as np
import logging

from utils import distilbert
from utils.hierarchy import PerLevelHierarchy


class CustomDataset(torch.utils.data.Dataset):
    """A PyTorch-compatible dataset class with hierarchical capabilities."""

    def __init__(
            self,
            df,
            hierarchy,
            tokenizer,
            max_len,

    ):
        """Create a dataset wrapper from the specified data."""
        self.tokenizer = tokenizer
        self.text = df['name']  # All datasets coming from adapters use this.
        # Level sizes
        self.levels = hierarchy.levels
        # All datasets coming from adapters use this.
        self.labels = df['codes']
        self.level_offsets = hierarchy.level_offsets
        self.max_len = max_len

    def __len__(self):
        """Return the number of rows in the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        """Equivalent to data[index]."""
        text = str(self.text.iloc[index])
        text = " ".join(text.split())
        inputs = self.tokenizer(
            text,
            None,  # No text_pair
            add_special_tokens=True,  # CLS, SEP
            max_length=self.max_len,
            padding='max_length',
            truncation=True
            # BERT tokenisers return attention masks by default
        )
        result = {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(self.labels.loc[index], dtype=torch.long),
        }
        return result


def get_loaders(
        name,
        config,
        verbose=False
):
    """Create loaders from the specified Parquet dataset.

    Parameters
    ----------
    name: str
        Name of the intermediate dataset. In other words, it is the name of
        the dataset's folder in `../datasets`. This function only works with
        datasets generated from adapters.
    config: dict
        A dictionary of configuration parameters for the loading process.
        The following parameters must be provided:

        - ``device``: ``'cuda'`` or ``'cpu'``
        - ``train_minibatch_size``: Size of a minibatch while training.
        - ``val_test_minibatch_size``: Size of a minibatch in the validation and test phase.

    verbose: bool
        If true, print more information about the importing process, such as
        all detected classes and hierarchy information.

    Returns
    -------
    train_loader: torch.utils.DataLoader
    val_loader: torch.utils.DataLoader
    test_loader: torch.utils.DataLoader
    hierarchy: PerLevelHierarchy

    See also
    --------
    utils.hierarchy.PerLevelHierarchy : the hierarchy class PyTorch models use.
    """
    train_path = 'datasets/{}/train.parquet'.format(name)
    val_path = 'datasets/{}/val.parquet'.format(name)
    test_path = 'datasets/{}/test.parquet'.format(name)

    targets = []
    if not os.path.exists(train_path):
        if not os.path.exists(train_path + '.dvc'):
            raise OSError('Training set not present and cannot be retrieved.')
        targets.append(train_path + '.dvc')

    if not os.path.exists(val_path):
        if not os.path.exists(val_path + '.dvc'):
            raise OSError(
                'Validation set not present and cannot be retrieved.')
        targets.append(val_path + '.dvc')

    if not os.path.exists(test_path):
        if not os.path.exists(train_path + '.dvc'):
            raise OSError('Test set not present and cannot be retrieved.')
        targets.append(test_path + '.dvc')

    if len(targets) > 0:
        os.system('dvc checkout {} {}'.format(
            ' '.join(targets), '-v' if verbose else ''))

    train = pd.read_parquet(train_path)
    val = pd.read_parquet(val_path)
    test = pd.read_parquet(test_path)
    hierarchy = PerLevelHierarchy.from_json(
        'datasets/{}/hierarchy.json'.format(name)).to(config['device'])

    # Pack into DataLoaders using CustomDataset instances
    train_loader = torch.utils.data.DataLoader(
        dataset=CustomDataset(
            train,
            hierarchy,
            distilbert.get_tokenizer(),
            64,
        ),
        batch_size=config['train_minibatch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=CustomDataset(
            val,
            hierarchy,
            distilbert.tokenizer,
            64,
        ),
        batch_size=config['val_test_minibatch_size'],
        shuffle=True,
        num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=CustomDataset(
            test,
            hierarchy,
            distilbert.tokenizer,
            64,
        ),
        batch_size=config['val_test_minibatch_size'],
        shuffle=True,
        num_workers=0
    )

    return train_loader, val_loader, test_loader, hierarchy


def get_hierarchical_one_hot(labels, level_sizes):
    """Binarise local-space labels to global-space one-hot vectors.

    This function is useful for models that use loss functions such as BCE.

    Parameters
    ----------
    labels: torch.Tensor or numpy.ndarray of shape (minibatch, depth)
        The local integer indices of true labels. For example,
        `[[0], [0]]` means the first label of each hierarchical level.
    level_sizes: list-like object of shape (depth)
        A list of level sizes. A level's size is the number of unique labels in
        that level.

    Returns
    -------
    binarised_labels: torch.LongTensor of shape `(minibatch, sum(level_sizes))`
        Per-level one-hot-encoded labels. Level encodings are concatenated
        to create a global binary label vector from the first level to the leaf.

    Examples
    --------
    >>> get_hierarchical_one_hot([[3, 5], [2, 0]], [4, 6])
    Tensor([[0,0,0,1, 0,0,0,0,0,1], [0,0,1,0, 1,0,0,0,0,0]])
    """
    return torch.cat(
        [
            torch.nn.functional.one_hot(
                label_level,
                num_classes=level_sizes[i]
            )
            for i, label_level in enumerate([
                    labels[:, i] for i in range(labels.shape[1])
            ])
        ],
        dim=1
    )


def get_metrics(test_output, display=None, compute_auprc=False):
    """Compute metrics for general PyTorch-based hierarchial models.

    The following metrics are computed:

    - Leaf accuracy (accuracy at the leaf level)
    - Leaf precision (precision at the leaf level)
    - Global accuracy (averaged accuracy over all levels)
    - Global precision (averaged precision over all levels)
    - (optionally) AU(PRC) (at the leaf level)

    Parameters
    ----------
    test_output: dict
        A dict containing the following keys:

        - ``outputs``: [torch.Tensor]
            List of Torch tensors, each represeting scores for a hierarchical level.
            A Tensor at index ``i`` of this list contains classification scores for
            each class in level ``i`` of the hierarchy and thus has shape
            (minibatch, level_sizes[i]).
        - ``targets``: torch.LongTensor of shape (minibatch, depth)
            List of integer label indices, ordered in hierarchical order (top to
            bottom). This can be taken straight from the 'codes' column of loaded
            Datasets.

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
        The list of metrics computed in the order listed above.

    """
    if test_output['targets'].ndim > 1:
        if test_output['targets'].shape[1] > 1:
            return get_hierarchical_metrics(test_output, display, compute_auprc)
        raise RuntimeError('Invalid array dimensionality: hierarchical models must return at least two levels.')
    return get_leaf_metrics(test_output, display, compute_auprc)


def get_leaf_metrics(test_output, display=None, compute_auprc=False):
    local_outputs = test_output['outputs']
    targets = test_output['targets']
    leaf_size = local_outputs.shape[1]

    def generate_one_hot(idx):
        b = np.zeros(leaf_size, dtype=bool)
        b[idx] = 1
        return b

    # Get predicted class indices at each level
    level_codes = np.argmax(local_outputs, axis=1)

    accuracy = metrics.accuracy_score(targets, level_codes)
    precision = metrics.precision_score(
        targets, level_codes, average='weighted', zero_division=0)

    if display == 'log' or display == 'both':
        logging.info('Leaf level:')
        logging.info("Accuracy: {}".format(accuracy))
        logging.info("Precision: {}".format(precision))

    if display == 'print' or display == 'both':
        print('Leaf level:')
        print("Accuracy: {}".format(accuracy))
        print("Precision: {}".format(precision))

    if compute_auprc:
        binarised_targets = np.array([
            generate_one_hot(idx) for idx in targets
        ])
        rectified_outputs = np.concatenate(
            [local_outputs, np.ones((1, leaf_size))],
            axis=0)
        rectified_targets = np.concatenate(
            [binarised_targets, np.ones((1, leaf_size), dtype=bool)],
            axis=0
        )

        auprc_score = metrics.average_precision_score(
            rectified_targets,
            rectified_outputs
        )
        if display == 'log':
            logging.info('Rectified leaf-level AU(PRC) score: {}'.format(
                auprc_score
            ))
        elif display == 'print':
            print('Rectified leaf-level AU(PRC) score: {}'.format(auprc_score))

        return np.array([accuracy, precision, None, None, auprc_score])
    return np.array([accuracy, precision, None, None])



def get_hierarchical_metrics(test_output, display=None, compute_auprc=False):
    local_outputs = test_output['outputs']
    targets = test_output['targets']
    leaf_size = local_outputs[-1].shape[1]
    depth = len(local_outputs)

    def generate_one_hot(idx):
        b = np.zeros(leaf_size, dtype=bool)
        b[idx] = 1
        return b

    # Get predicted class indices at each level
    level_codes = [
        np.argmax(local_outputs[level], axis=1)
        for level in range(len(local_outputs))
    ]

    accuracies = [
        metrics.accuracy_score(
            targets[:, level], level_codes[level]
        ) for level in range(depth)
    ]
    precisions = [
        metrics.precision_score(
            targets[:, level], level_codes[level], average='weighted',
            zero_division=0
        ) for level in range(depth)
    ]

    global_accuracy = sum(accuracies)/len(accuracies)
    global_precision = sum(precisions)/len(precisions)

    if display == 'log' or display == 'both':
        for i in range(depth):
            logging.info('Level {}:'.format(i))
            logging.info("Accuracy: {}".format(accuracies[i]))
            logging.info("Precision: {}".format(precisions[i]))
        logging.info('Global level:')
        logging.info("Accuracy: {}".format(global_accuracy))
        logging.info("Precision: {}".format(global_precision))
    if display == 'print' or display == 'both':
        for i in range(depth):
            print('Level {}:'.format(i))
            print("Accuracy: {}".format(accuracies[i]))
            print("Precision: {}".format(precisions[i]))
        print('Global level:')
        print("Accuracy: {}".format(global_accuracy))
        print("Precision: {}".format(global_precision))

    if compute_auprc:
        binarised_targets = np.array([
            generate_one_hot(lst[-1]) for lst in targets
        ])
        rectified_outputs = np.concatenate(
            [local_outputs[-1], np.ones((1, local_outputs[-1].shape[1]))],
            axis=0
        )
        rectified_targets = np.concatenate([binarised_targets, np.ones((1, leaf_size), dtype=bool)], axis=0)

        auprc_score = metrics.average_precision_score(rectified_targets, rectified_outputs)
        if display == 'log':
            logging.info('Rectified leaf-level AU(PRC) score: {}'.format(auprc_score))
        elif display == 'print':
            print('Rectified leaf-level AU(PRC) score: {}'.format(auprc_score))

        return np.array([accuracies[-1], precisions[-1], global_accuracy, global_precision, auprc_score])
    return np.array([accuracies[-1], precisions[-1], global_accuracy, global_precision])
