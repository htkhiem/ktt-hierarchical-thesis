"""
Dataset logic for models.

This file contains the following:
- PyTorch data loader utilities for generating a PyTorch-compatible dataset
handler.
- Universal label parsing functions to vectorise and binarise textual class
names.
"""
import numpy as np
import torch
import pandas as pd


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
            distilbert.tokenizer,
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
