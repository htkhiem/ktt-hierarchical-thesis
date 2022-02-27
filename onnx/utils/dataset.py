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

# CLASS NAME ENCODING ---------------------------------------------------------


def preprocess_classes(data, original_name, depth, verbose=False):
    """
    Build a list of unique class names for each level and create bidirectional mappings.
    """
    cls2idx = []
    idx2cls = []
    for i in range(depth):
        category_li = data[original_name].apply(
            lambda lst: lst[i]
        ).astype('category')
        if verbose:
            print(category_li.cat.classes)
        cls2idx.append(dict([
            (category, index)
            for (index, category)
            in enumerate(category_li.cat.categories)
        ]))
        idx2cls.append(list(category_li.cat.categories))
    return cls2idx, idx2cls


def class_to_index(data, original_name, cls2idx, depth):
    """Build name-to-index mappings."""
    data['codes'] = data[original_name].apply(
        lambda lst: [
            cls2idx[i][cat]
            for (i, cat)
            in enumerate(lst[:depth])
        ],
    ).astype('object')


def index_to_binary(data, index_col_name, offsets, sz, verbose=False):
    """Build binary vectors for class membership."""
    if verbose:
        print('Using offsets:', offsets)

    def generate_binary(codes):
        b = np.zeros(sz, dtype=int)
        indices = np.array(codes, dtype=int) + offsets[:-1]
        if verbose:
            print(codes, offsets, indices)
        b[indices] = 1
        return b.tolist()

    data[index_col_name + '_b'] = data[index_col_name].apply(
        lambda lst: generate_binary(lst),
    )

# DATASETS & DATALOADERS CLASSES ----------------------------------------------
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, hierarchy, tokenizer, max_len, text_col_name='title'):
        self.tokenizer = tokenizer
        self.text = df[text_col_name]
        # Level sizes
        self.levels = hierarchy.levels
        self.labels = df.codes
        self.labels_b = df.codes_b
        self.level_offsets = hierarchy.level_offsets
        self.max_len = max_len


    def __len__(self):
        return len(self.text)


    def __getitem__(self, index):
        text = str(self.text.iloc[index])
        text = " ".join(text.split())
        inputs = self.tokenizer(
            text,
            None, # No text_pair
            add_special_tokens=True, # CLS, SEP
            max_length=self.max_len, # For us it's a hyperparam. See next cells.
            padding='max_length',
            truncation=True
            # BERT tokenisers return attention masks by default
        )
        result = {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(self.labels.loc[index], dtype=torch.long),
            'labels_b': torch.tensor(self.labels_b.iloc[index], dtype=torch.long)
        }
        return result

# OVERALL SCRIPT ---------------------------------------------------------------------------------------------------------------
# Values used for published results


RANDOM_SEED = 123
TRAIN_SET_RATIO = 0.8
VAL_SET_RATIO = 0.1


def get_loaders(
        path,
        config,
        depth=2,
        binary=True,
        build_parent=True,
        build_R=False,
        build_M=False,
        full_set=True,
        input_col_name='title',
        class_col_name='category',
        partial_set_frac=0.05,
        verbose=False
):
    """Create loaders from the specified Parquet dataset."""
    # path: Relative path (from main.py) to dataset .parquet (may be a folder
    #   in case it's partitioned)
    # depth: Hierarchical depth to traverse
    # binary: Whether to genenrate binary vector encodings or not (for C-HMCNN
    #   and DB-FBHCN+AWX)
    # build_parent: Whether to build parent_of (for HMCN-F and both DB-BHCN
    #   variants)
    # build_R: Whether to build the R-matrix (for DB-BHCN+AWX)
    # build_M: Whether to build the M-matrix (for C-HMCNN)
    data = pd.read_parquet(path)
    cls2idx, idx2cls = preprocess_classes(data, class_col_name, depth)
    class_to_index(data, class_col_name, cls2idx, depth)
    # Generate hierarchy
    hierarchy = PerLevelHierarchy(
        data['codes'],
        cls2idx,
        build_parent,
        build_R,
        build_M
    ).to(config['device'])
    if binary:
        index_to_binary(data, 'codes', hierarchy.level_offsets, len(hierarchy.classes), verbose)
        columns = [input_col_name, 'codes', 'codes_b']
    else:
        columns = [input_col_name, 'codes']

    # CV-splitting
    filtered = None
    train_set = None
    test_set = None
    if not full_set:
        filtered = data.sample(frac = partial_set_frac, random_state=RANDOM_SEED)[columns]
    else:
        filtered = data[columns]

    train_set = filtered.sample(frac = TRAIN_SET_RATIO, random_state=RANDOM_SEED)
    val_test_set = filtered.drop(train_set.index)

    val_set = val_test_set.sample(frac = VAL_SET_RATIO / (1-TRAIN_SET_RATIO), random_state=RANDOM_SEED)
    test_set = val_test_set.drop(val_set.index)

    train_set = train_set.reset_index(drop=True)
    val_set = val_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)

    train_set = CustomDataset(train_set, hierarchy, distilbert.tokenizer, 64, text_col_name=input_col_name)
    val_set = CustomDataset(val_set, hierarchy, distilbert.tokenizer, 64, text_col_name=input_col_name)
    test_set = CustomDataset(test_set, hierarchy, distilbert.tokenizer, 64, text_col_name=input_col_name)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=config['train_minibatch_size'], shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=config['val_test_minibatch_size'], shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=config['val_test_minibatch_size'], shuffle=True, num_workers=0)

    return train_loader, val_loader, test_loader, hierarchy
