import argparse
import logging
import json
import numpy as np
import os, glob
import torch
from functools import partial

# from models import db_bhcn, db_ahmcnf, db_achmcnn, tfidf_hsgd
from utils import dataset, distilbert
from utils.export import export_classifier, export_distilbert


def get_latest_checkpoint(model_name, dataset_name):
    """Grabs latest checkpoint in the automatic folder structure."""
    weight_names = sorted(glob.glob('weights/{}/{}/run_*.pt'.format(model_name, dataset_name)))
    weight_name = weight_names[-1]  # For now default to the latest weight
    return torch.load(weight_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--retrieve', action='store_true', help='Retrieve additional datasets (five Amazon metadata sets).')
    parser.add_argument('-n', '--dry_run', action='store_true', help='Don\'t save trained weights. Results are still logged to the logfile. Useful for when you run low on disk space.')
    parser.add_argument('-d', '--dataset', help='Pass a comma-separated list of dataset names (excluding .parquet) to use. This is a required argument.')
    parser.add_argument('-D', '--distilbert', action='store_true', help='If this flag is specified, download DistilBERT pretrained weights from huggingface to your user temp directory. By default, this repository tries to look for an offline-cached version instead.')
    parser.add_argument('-m', '--model', help="""Pass a comma-separated list of model names to run. Available models:
\tdb_bhcn\t\t(DistilBERT Branching Hierarchical Classifier)
\tdb_bhcn_awx\t\t(DistilBERT Branching Hierarchical Classifier + Adjacency Wrapping Matrix)
\tdb_ahmcnf\t\t(Adapted HMCN-F model running on DistilBERT encodings)
\tdb_achmcnn\t\t(Adapted C-HMCNN model running on DistilBERT encodings)
\ttfidf_hsgd\t\t(Internal-node SGD classifier hierarchy using tf-idf encodings)
By default, all models are run.""")
    parser.add_argument('-b', '--bento', action='store_true', help='Add exported models to the local BentoML model store.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print more information to the console (for debugging purposes).')
    parser.add_argument('-c', '--cpu', action='store_true', help='Only run on CPU. Use this if you have to run without CUDA support (warning: depressingly slow).')

    args = parser.parse_args()

    # Defaults
    verbose = False
    bento = False
    with open('./hyperparams.json', 'r') as j:
        hyperparams = json.loads(j.read())
    model_lst = [
        'db_bhcn',
        'db_bhcn_awx',
        'db_ahmcnf',
        'db_achmcnn',
        'tfidf_hsgd'
    ]

    dataset_lst = [name.strip() for name in args.dataset.split(",")]

    if args.model:
        model_lst = [name.strip() for name in args.model.split(",")]

    if args.verbose:
        verbose = args.verbose

    if args.bento:
        bento = args.bento

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print('Using', device)

    distilbert.init()

    # Specific requirements from each model
    build_R = 'db_bhcn_awx' in model_lst
    build_M = 'db_achmcnn' in model_lst

    for dataset_name in dataset_lst:
        config = {
            'device': device,
            'dataset_name': dataset_name,
            'train_minibatch_size': 1,
            'val_test_minibatch_size': 1,
        }
        # Generate enough hierarchical data for all selected models
        _, _, _, hierarchy = dataset.get_loaders(
            '../datasets/{}.parquet'.format(dataset_name),
            config,
            full_set=True,
            build_R=build_R,
            build_M=build_M,
            verbose=verbose,
        )
        if 'db_bhcn' in model_lst:
            MODEL_NAME = 'db_bhcn'
            checkpoint = get_latest_checkpoint(MODEL_NAME, dataset_name)
            # Export DistilBERT with finetuned weights
            export_distilbert(
                dataset_name,
                MODEL_NAME,
                checkpoint['encoder_state_dict'],
                bento
            )
            config = hyperparams[MODEL_NAME]
            config['device'] = device
            config['dataset_name'] = dataset_name
            config['bento'] = bento
            export_classifier(
                checkpoint['classifier_state_dict'],
                MODEL_NAME,
                dataset_name,
                config,
                hierarchy
            )

        if 'db_bhcn_awx' in model_lst:
            MODEL_NAME = 'db_bhcn_awx'
            checkpoint = get_latest_checkpoint(MODEL_NAME, dataset_name)
            # Export DistilBERT with finetuned weights
            export_distilbert(
                dataset_name,
                MODEL_NAME,
                checkpoint['encoder_state_dict'],
                bento
            )
            config = hyperparams[MODEL_NAME]
            config['device'] = device
            config['dataset_name'] = dataset_name
            config['bento'] = bento
            export_classifier(
                checkpoint['classifier_state_dict'],
                MODEL_NAME,
                dataset_name,
                config,
                hierarchy
            )
        if 'db_ahmcnf' in model_lst:
            MODEL_NAME = 'db_ahmcnf'
            checkpoint = get_latest_checkpoint(MODEL_NAME, dataset_name)
            # Export DistilBERT with finetuned weights
            export_distilbert(
                dataset_name,
                MODEL_NAME,
                bento=bento
                # No state dict - AHMCN-F doesn't like finetuning
            )
            config = hyperparams[MODEL_NAME]
            config['device'] = device
            config['dataset_name'] = dataset_name
            config['bento'] = bento
            export_classifier(
                checkpoint['state_dict'],
                MODEL_NAME,
                dataset_name,
                config,
                hierarchy
            )

        if 'db_achmcnn' in model_lst:
            MODEL_NAME = 'db_achmcnn'
            checkpoint = get_latest_checkpoint(MODEL_NAME, dataset_name)
            # Export DistilBERT with finetuned weights
            export_distilbert(
                dataset_name,
                MODEL_NAME,
                checkpoint['encoder_state_dict'],
                bento
            )
            config = hyperparams[MODEL_NAME]
            config['device'] = device
            config['dataset_name'] = dataset_name
            config['bento'] = bento
            export_classifier(
                checkpoint['classifier_state_dict'],
                MODEL_NAME,
                dataset_name,
                config,
                hierarchy
            )
