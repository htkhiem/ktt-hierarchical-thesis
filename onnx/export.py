"""
Model export controller.

This script controls the exporting of models, either directly to ONNX,
or to a BentoML ModelStorage.
"""
import argparse
import logging
import json
import numpy as np
import os, glob
import torch
from functools import partial

from utils import dataset, distilbert
from models import db_bhcn, db_ahmcnf, db_achmcnn, db_linear, tfidf_hsgd


def get_path(model_name, dataset_name, best=True, idx=None):
    """
    Grabs latest checkpoint in the automatic folder structure.

    If best is True, get paths for 'best' weights instead of last-epoch.
    If idx is specified, get that index specifically. If not, get the latest.
    """
    weight_names = sorted(glob.glob(
        'weights/{}/{}/run_{}{}.pt'.format(
            model_name,
            dataset_name,
            str(idx) if idx is not None else '*',
            '_best' if best else ''
        )
    ))
    return weight_names[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='Pass a comma-separated list of dataset names (excluding .parquet) to use. This is a required argument.')
    parser.add_argument('-m', '--model', help="""Pass a comma-separated list of model names to run. Available models:
\tdb_bhcn\t\t(DistilBERT Branching Hierarchical Classifier)
\tdb_bhcn_awx\t\t(DistilBERT Branching Hierarchical Classifier + Adjacency Wrapping Matrix)
\tdb_ahmcnf\t\t(Adapted HMCN-F model running on DistilBERT encodings)
\tdb_achmcnn\t\t(Adapted C-HMCNN model running on DistilBERT encodings)
\tdb_linear\t\t(DistilBERT + Linear layer)
\ttfidf_hsgd\t\t(Internal-node SGD classifier hierarchy using tf-idf encodings)
By default, all models are exported. An error will be raised if a model has not been trained with any of the specified datasets.""")
    parser.add_argument('-i', '--index', help='Optionally specify which trained weights to load by their indices.')
    parser.add_argument('-B', '--best', action='store_true', help='User best-epoch weights instead of latest-epoch.')
    parser.add_argument('-b', '--bento', action='store_true', help='Add exported models to the local BentoML model store.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print more information to the console (for debugging purposes).')
    parser.add_argument('-c', '--cpu', action='store_true', help='Only trace models using CPU. Use this if you have to run without CUDA support.')

    args = parser.parse_args()

    # Defaults
    verbose = False
    bento = False
    best = False
    index = None
    with open('./hyperparams.json', 'r') as j:
        hyperparams = json.loads(j.read())
    model_lst = [
        'db_bhcn',
        'db_bhcn_awx',
        'db_ahmcnf',
        'db_achmcnn',
        'db_linear',
        'tfidf_hsgd'
    ]

    dataset_lst = [name.strip() for name in args.dataset.split(",")]

    if args.model:
        model_lst = [name.strip() for name in args.model.split(",")]

    if args.verbose:
        verbose = args.verbose

    if args.bento:
        bento = args.bento

    if args.index:
        index = args.index

    if args.best:
        best = True

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print('Using', device)

    distilbert.init()

    for dataset_name in dataset_lst:
        if 'db_bhcn' in model_lst:
            model = db_bhcn.DB_BHCN.from_checkpoint(
                get_path('db_bhcn', dataset_name, best, index)
            ).to(device)
            model.export(dataset_name, bento)

        if 'db_bhcn_awx' in model_lst:
            model = db_bhcn.DB_BHCN.from_checkpoint(
                get_path('db_bhcn_awx', dataset_name, best, index)
            ).to(device)
            model.export(dataset_name, bento)

        if 'db_ahmcnf' in model_lst:
            model = db_ahmcnf.DB_AHMCN_F.from_checkpoint(
                get_path('db_ahmcnf', dataset_name, best, index)
            ).to(device)
            model.export(dataset_name, bento)

        if 'db_achmcnn' in model_lst:
            model = db_achmcnn.DB_AC_HMCNN.from_checkpoint(
                get_path('db_achmcnn', dataset_name, best, index)
            ).to(device)
            model.export(dataset_name, bento)

        if 'db_linear' in model_lst:
            model = db_linear.DB_Linear.from_checkpoint(
                get_path('db_linear', dataset_name, best, index)
            ).to(device)
            model.export(dataset_name, bento)

        if 'tfidf_hsgd' in model_lst:
            model = tfidf_hsgd.Tfidf_HSGD.from_checkpoint(
                get_path('tfidf_hsgd', dataset_name, best, index)
            )
            model.export(dataset_name, bento)
