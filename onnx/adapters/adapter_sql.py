"""
SQL data adapter.

This adapter accepts a database connection and two queries. It expects the two queries
to return two views in the following schema:

View DATAITEM:
id: int
name: varchar
class_id: int

View CLASS:
id: int
parent_id: int
"""
import sys
sys.path.append('../')

import os
import json
import argparse
import psycopg2
import pandas as pd
import numpy as np
from queue import Queue
from collections import namedtuple
from utils.hierarchy import PerLevelHierarchy

# Readable definition of a Node in the hierarchical DAG.
Node = namedtuple('Node', ['db_id', 'parent_db_id'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        help='Path to configuration file. Defaults to adapter_sql.cfg in current directory.'
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

    config_path = './adapter_sql.json' if not args.config else args.config
    dataset_name = args.title
    verbose = False if not args.verbose else args.verbose
    train_ratio = 0.9 if not args.train else args.train
    val_ratio = 0.5 if not args.validate else args.validate
    seed = 0 if not args.seed else args.seed

    assert(train_ratio > 0 and train_ratio < 1)
    assert(val_ratio > 0 and val_ratio < 1)

    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
        print(config)

    conn = psycopg2.connect(
        host=config['host'],
        database=config['database'],
        user=config['user'],
        password=config['password']
    )

    sql_dataitems = pd.io.sql.read_sql(config['dataitem_query'], conn)
    sql_classes = pd.io.sql.read_sql(config['class_query'], conn)

    # Build the tree. Root node has no name and  ID = -1 to completely avoid
    # collision risks
    root_node = Node(-1, -1)

    # In the view, parent-less categories have NaN for parent_id. We can
    # replace them with -1 (the special internal ID we use for root_note
    # above).
    # This avoids an if-statement and helps with straight-line execution
    # performance.
    sql_classes['parent_id'].fillna(-1, inplace=True)
    sql_classes.reset_index(inplace=True)
    # We might be stuck with a for-loop, as the body uses code with side
    # effects.
    # We must use some sort of storage as we cannot be guaranteed to be able to
    # iterate through the table in a hierarchical manner.
    # As such, this node dict allows for easy access to all nodes at once, no
    # matter the iteration order.
    node_dict = {
        -1: root_node
    }
    # Auxillary dict to keep track of child nodes. This is stored externally as
    # we would often reach a child node before the parent node is discovered.
    children_dict = {
        -1: []
    }

    # Build node_dict and children_dict
    for row in sql_classes.itertuples():
        node = Node(row[2], int(row[3]))
        node_dict[row[2]] = node
        try:
            children_dict[row[3]].append(node)
        except KeyError:
            children_dict[row[3]] = [node]

    if verbose:
        print('Building preliminary data structures...')

    classes = []  # Essentially idx2id
    id2idx = {}  # A mapping from database ID to internal indices
    level_offsets = [0]
    level_sizes = []

    even_queue = Queue()
    odd_queue = Queue()

    for node in children_dict[-1]:
        odd_queue.put(node)

    depth = 1  # Which queue to pop depends on depth

    put_queue = even_queue
    pop_queue = odd_queue

    while not even_queue.empty() or not odd_queue.empty():
        if verbose:
            print('Level', depth)
        level_sizes.append(pop_queue.qsize())
        while not pop_queue.empty():
            node = pop_queue.get()
            idx = len(classes)
            classes.append(node.db_id)
            id2idx[node.db_id] = idx
            if verbose:
                print('Node ID {} mapped to internal index {}'.format(
                    node.db_id, idx
                ))
            try:
                for child in children_dict[node.db_id]:
                    put_queue.put(child)
            except KeyError:
                # Node has no children
                pass

        level_offsets.append(len(classes))

        # Flip queues
        print(classes)
        temp = pop_queue
        pop_queue = put_queue
        put_queue = temp
        depth += 1

    # Special case: root node
    id2idx[-1] = -1

    if verbose:
        print('Building parent_of...')

    even_queue = Queue()
    odd_queue = Queue()

    for node in children_dict[-1]:
        odd_queue.put(node)

    depth = 1  # which queue to pop depends on depth

    put_queue = even_queue
    pop_queue = odd_queue

    parent_of = []

    while not even_queue.empty() or not odd_queue.empty():
        if verbose:
            print('Level', depth)
        parent_of.append([-1] * pop_queue.qsize())
        while not pop_queue.empty():
            node = pop_queue.get()
            child_idx = id2idx[node.db_id]
            local_child_idx = child_idx - level_offsets[depth-1]
            parent_idx = id2idx[node.parent_db_id]
            local_parent_idx = parent_idx - level_offsets[depth-2] if parent_idx != -1 else child_idx
            if verbose:
                print('Child idx: {}; parent idx: {}; local parent idx: {}'.format(child_idx, parent_idx, local_parent_idx))

            # Actually writing to parent_of
            parent_of[-1][local_child_idx] = local_parent_idx

            try:
                for child in children_dict[node.db_id]:
                    put_queue.put(child)
            except KeyError:
                # Node has no children
                pass

        # Flip queues
        temp = pop_queue
        pop_queue = put_queue
        put_queue = temp
        depth += 1

    def trace_hierarchy(leaf_db_id):
        """Given leaf label's database ID, return hierarchical path from top level but in internal indices."""
        path = np.array([id2idx[leaf_db_id]])
        parent_db_id = node_dict[leaf_db_id].parent_db_id
        while parent_db_id != -1:
            path = np.append(path, [id2idx[parent_db_id]])
            parent_db_id = node_dict[parent_db_id].parent_db_id
        return np.flip(path) - level_offsets[:-1]

    sql_dataitems['codes'] = sql_dataitems['class_id'].apply(trace_hierarchy)

    hierarchy = PerLevelHierarchy(
        codes=sql_dataitems['codes'],
        cls2idx=id2idx,  # Use database keys instead of textual names.
        levels=level_sizes,
        level_offsets=level_offsets,
        classes=classes,
        build_parent=True,
        build_R=True,
        build_M=True
    )

    path = '../datasets/{}/'.format(dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)

    # Output dataset as parquets
    sql_dataitems = sql_dataitems[['id', 'name', 'codes']]

    train_set = sql_dataitems.sample(frac=train_ratio, random_state=seed)
    sql_dataitems = sql_dataitems.drop(train_set.index)

    val_set = sql_dataitems.sample(frac=val_ratio, random_state=seed)
    test_set = sql_dataitems.drop(val_set.index)

    train_set = train_set.reset_index(drop=True)
    val_set = val_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)

    train_set.to_parquet(path + 'train.parquet')
    val_set.to_parquet(path + 'val.parquet')
    test_set.to_parquet(path + 'test.parquet')

    hierarchy.to_json(path + 'hierarchy.json')

    print('Finished!')
