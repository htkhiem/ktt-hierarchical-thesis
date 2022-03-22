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
import pathlib
import argparse
import pandas as pd
from utils.hierarchy import PerLevelHierarchy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--path',
        required=True,
        help='Path to the flat file.'
    )
    parser.add_argument(
        '-N',
        '--name',
        help='Title of the name column. Defaults to \'name\'.'
    )
    parser.add_argument(
        '-C',
        '--classes',
        help='Title of the classes column. Defaults to \'classes\'.'
    )
    parser.add_argument(
        '-d',
        '--depth',
        help='Maximum depth to build hierarchy. Defaults to 4.'
    )
    parser.add_argument(
        '-j',
        '--json',
        help='Manually specify the JSON schema, if the file is in JSON format. See pandas read_json documentation. Default to \'records\''
    )
    parser.add_argument(
        '-T',
        '--train',
        help='How much to use as training set when CV-splitting SQL data. Default to 0.9. Range: (0, 1).'
    )
    parser.add_argument(
        '-V',
        '--validate',
        help='How much of the remaining set to use for validation set when CV-splitting SQL data. Default to 0.5. Range: (0, 1).'
    )
    parser.add_argument(
        '-s',
        '--seed',
        help='Seed used for random sampling in CV-splitting. Default to 0.'
    )
    parser.add_argument(
        '-P',
        '--proportion',
        help='How much of the dataset to actually output. Defaults to 1.0. Range: (0.0, 1.0].'
    )
    parser.add_argument(
        '-t',
        '--title',
        required=True,
        help='Name of dataset to generate.'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Path to configuration file. Defaults to adapter_sql.cfg in current directory.'
    )

    args = parser.parse_args()

    path = args.path
    name_col = 'name' if not args.name else args.name
    depth = 4 if not args.depth else int(args.depth)
    proportion = 1.0 if not args.proportion else float(args.proportion)
    classes_col = 'classes' if not args.classes else args.classes
    dataset_name = args.title
    json_format = 'records' if not args.json else args.json
    verbose = False if not args.verbose else args.verbose
    train_ratio = 0.9 if not args.train else float(args.train)
    val_ratio = 0.5 if not args.validate else float(args.validate)
    seed = 0 if not args.seed else args.seed

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
            name_col: 'name',
            classes_col: 'classes'
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

    path = '../datasets/{}/'.format(dataset_name)
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

    print('Finished!')
