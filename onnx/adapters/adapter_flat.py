"""
Flatfile data adapter.

This adapter accepts a flat file (JSON, CSV or Parquet/Feather) with at least the following
columns:
- name: string
- classes: list of classes, hierarchically ordered from root to leaf, that
  this data item belongs to.
The actual column names can be specified via arguments.
"""

import sys
sys.path.append('../')

import os
import click
from textwrap import dedent
import pathlib
import argparse
import pandas as pd
from utils.hierarchy import PerLevelHierarchy


@click.command()
@click.argument('path', required=1)
@click.argument('name', required=1)
@click.option(
    '-t',
    '--text',
    default='text',
    show_default=True,
    help='Title of the input column.'
)
@click.option(
    '-c',
    '--classes',
    default='classes',
    show_default=True,
    help='Title of the classes (labels) column.'
)
@click.option(
    '-d',
    '--depth',
    default=2,
    show_default=True,
    help='Maximum depth to build hierarchy.'
)
@click.option(
    '-j',
    '--json-format',
    default='records',
    show_default=True,
    help=dedent("""
    Manually specify the JSON schema, if the file is in JSON format. See pandas
    read_json documentation.
    """)
)
@click.option(
    '--train-ratio',
    default=0.9,
    show_default=True,
    help=dedent("""
    How much to use as training set when CV-splitting SQL data. Range: (0, 1).
    """)
)
@click.option(
    '--val-ratio',
    default=0.5,
    show_default=True,
    help=dedent("""
    How much of the remaining set to use for validation set when CV-splitting
    data. Range: (0, 1).
    """)
)
@click.option(
    '--seed',
    default=0,
    show_default=True,
    help=dedent("""
    Seed used for random sampling in CV-splitting.
    """)
)
@click.option(
    '-p',
    '--proportion',
    default=1.0,
    show_default=True,
    help=dedent("""
    How much of the dataset to actually output. Range: (0.0, 1.0].
    """)
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    default=False,
    help=dedent("""
    Verbose mode (print more information about the process).
    """)
)
@click.option(
    '--dvc',
    is_flag=True,
    default=True,
    show_default=True,
    help=dedent("""
    Track this dataset using DVC.
    """)
)
def main(
        path,
        name,
        text,
        classes,
        depth,
        json_format,
        train_ratio,
        val_ratio,
        seed,
        proportion,
        verbose,
        dvc
):
    """Adapt a flat file (CSV/JSON/Arrow/...) into the intermediate schema.

    It takes in a PATH to the flat file and outputs an intermediate dataset
    named NAME.
    """
    assert(train_ratio > 0 and train_ratio < 1)
    assert(val_ratio > 0 and val_ratio < 1)
    assert(json_format in ['split', 'records', 'index', 'columns', 'values'])

    format = pathlib.Path(path).suffix
    if format == '.json':
        data = pd.read_json(
            path,
            orient=json_format,
        )
    elif format == '.csv':
        data = pd.read_csv(
            path
        )
    elif format == '.parquet':
        data = pd.read_parquet(path)
    elif format == '.feather':
        data = pd.read_feather(path)
    else:
        raise RuntimeError('Unsupported flatfile format!')

    # Rename columns back to standard internal schema
    data.rename(
        columns={
            text: 'name',
            classes: 'classes'
        }, inplace=True
    )

    # Trim to maximum depth
    data['classes'] = data['classes'].apply(
        lambda lst: lst[:min(depth, len(lst))]
    )

    # Generate class-to-index mapping.
    cls2idx = []
    for i in range(depth):
        category_li = data['classes'].apply(
            lambda lst: lst[i]
        ).astype('category')
        if verbose:
            print(category_li.cat.categories)
        cls2idx.append(dict([
            (category, index)
            for (index, category)
            in enumerate(category_li.cat.categories)
        ]))

    # Write the class index columns
    data['codes'] = data['classes'].apply(
        lambda lst: [
            cls2idx[i][cat]
            for (i, cat)
            in enumerate(lst[:depth])
        ],
    ).astype('object')

    hierarchy = PerLevelHierarchy(
        codes=data['codes'],
        cls2idx=cls2idx,  # Use database keys instead of textual names.
        build_parent=True,
        build_R=True,
        build_M=True
    )

    path = '../datasets/{}/'.format(name)
    if not os.path.exists(path):
        os.makedirs(path)

    # Output dataset as parquets
    data = data[['name', 'codes']]

    if proportion != 1.0:
        data = data.sample(frac=proportion, random_state=seed)

    train_set = data.sample(frac=train_ratio, random_state=seed)
    data = data.drop(train_set.index)

    val_set = data.sample(frac=val_ratio, random_state=seed)
    test_set = data.drop(val_set.index)

    train_set = train_set.reset_index(drop=True)
    val_set = val_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)

    train_set.to_parquet(path + 'train.parquet')
    val_set.to_parquet(path + 'val.parquet')
    test_set.to_parquet(path + 'test.parquet')

    hierarchy.to_json(path + 'hierarchy.json')

    if dvc:
        os.system('dvc add {} {} {} {}'.format(
            path + 'train.parquet',
            path + 'val.parquet',
            path + 'test.parquet',
            path + 'hierarchy.json'
        ))

    print('Finished!')


if __name__ == '__main__':
    main()
