"""This file defines functions specific to PyTorch/DistilBERT models."""

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
    """
    Create loaders from the specified Parquet dataset.

    name: Dataset folder name in ../datasets.
    This function only works with datasets generated from adapters.
    """
    train = pd.read_parquet('../datasets/{}/train.parquet'.format(name))
    val = pd.read_parquet('../datasets/{}/val.parquet'.format(name))
    test = pd.read_parquet('../datasets/{}/test.parquet'.format(name))
    hierarchy = PerLevelHierarchy.from_json(
        '../datasets/{}/hierarchy.json'.format(name)).to(config['device'])

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
    """
    Binarise local-space labels to global-space one-hot vectors.

    Example input:
    - Labels: [[3, 5], [2, 0]], where axis 0 is minibatch and axis 1 is
      hierarchical depth.
    - Level sizes: [4, 6]

    Corresponding output:
    [[0,0,0,1, 0,0,0,0,0,1], [0,0,1,0, 1,0,0,0,0,0]], where axis 0 is
    minibatch and axis 1 is global label space.
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
    """
    Compute metrics for general PyTorch-based hierarchial models.

    local_outputs: list of Torch tensors, each represeting scores for a
    hierarchical level.
    targets: list of category codes, ordered in hierarchical order (top to
    bottom). This can be taken straight from the 'codes' column.
    """
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
