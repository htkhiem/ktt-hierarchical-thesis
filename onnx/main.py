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
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--retrieve', action='store_true', help='Retrieve additional datasets (five Amazon metadata sets).')
    parser.add_argument('-n', '--dry_run', action='store_true', help='Don\'t save trained weights. Results are still logged to the logfile. Useful for when you run low on disk space.')
    parser.add_argument('-d', '--dataset', help='Pass a comma-separated list of dataset names (excluding .parquet) to use. By default, all six datasets presented in the paper are used.')
    parser.add_argument('-D', '--distilbert', action='store_true', help='If this flag is specified, download DistilBERT pretrained weights from huggingface to your user temp directory. By default, this repository tries to look for an offline-cached version instead.')
    parser.add_argument('-m', '--model', help="""Pass a comma-separated list of model names to run. Available models:
\tdb_bhcn\t\t(DistilBERT Branching Hierarchical Classifier)
\tdb_bhcn_awx\t\t(DistilBERT Branching Hierarchical Classifier + Adjacency Wrapping Matrix)
\tdb_ahmcnf\t\t(Adapted HMCN-F model running on DistilBERT encodings)
\tdb_achmcnn\t\t(Adapted C-HMCNN model running on DistilBERT encodings)
\ttfidf_hsgd\t\t(Internal-node SGD classifier hierarchy using tf-idf encodings)
By default, all models are run.""")
    parser.add_argument('-v', '--verbose', action='store_true', help='Print more information to the console (for debugging purposes).')
    parser.add_argument('-c', '--cpu', action='store_true', help='Only run on CPU. Use this if you have to run without CUDA support (warning: depressingly slow).')

    args = parser.parse_args()

    # Defaults
    verbose = False
    with open('./hyperparams.json', 'r') as j:
        hyperparams = json.loads(j.read())
    urls = {
        'Arts_Crafts_and_Sewing': 'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles2/meta_Arts_Crafts_and_Sewing.json.gz',
        'Electronics': 'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles2/meta_Electronics.json.gz',
        'Grocery_and_Gourmet_Food': 'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles2/meta_Grocery_and_Gourmet_Food.json.gz',
        'Industrial_and_Scientific': 'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles2/meta_Industrial_and_Scientific.json.gz',
        'Musical_Instruments': 'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles2/meta_Musical_Instruments.json.gz',
        'Walmart_30k': 'offline'
    }
    dataset_lst = urls.keys()
    model_lst = [
        'db_bhcn',
        'db_bhcn_awx',
        'db_ahmcnf',
        'db_achmcnn',
        'tfidf_hsgd'
    ]

    if args.dataset:
        dataset_lst = [name.strip() for name in args.dataset.split(",")]

    if args.model:
        model_lst = [name.strip() for name in args.model.split(",")]

    if args.verbose:
        verbose = args.verbose

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print('Using', device)

    distilbert.init()

    for dataset_name in dataset_lst:
        if 'db_bhcn' in model_lst:
            weight_names = sorted(glob.glob('weights/db_bhcn/run_*.pt'))
            weight_name = weight_names[-1] # For now default to the latest weight
            checkpoint = torch.load(weight_name)
            # print(checkpoint['encoder_state_dict'])
            # Export DistilBERT with finetuned weights
            export_distilbert(checkpoint['encoder_state_dict'], dataset_name, 'db_bhcn')
            config = hyperparams['db_bhcn']
            config['device'] = device
            config['dataset_name'] = dataset_name
            _, _, _, hierarchy = dataset.get_loaders(
                '../datasets/{}.parquet'.format(dataset_name),
                config,
                full_set=True,
                verbose=verbose,
            )
            export_classifier(checkpoint['classifier_state_dict'], 'db_bhcn', dataset_name, config, hierarchy)
