"""
Training controller.

Use this Python script to initiate batched training of any combination of model
and dataset.
"""
import os
import argparse
import logging
import json
import numpy as np
import torch

import datetime
import matplotlib.pyplot as plt

from utils.metric import get_metrics, get_leaf_report
from models import db_bhcn, db_ahmcnf, db_achmcnn, tfidf_hsgd
from utils import dataset, distilbert


def train_val_test(
        config,
        model,
        train_loader,
        val_loader,
        test_loader,
        metrics_func=get_metrics,
        save_weights=True,
        verbose=False,
        balanced=False,
        report=False,
):
    """
    Train a model over a dataset.

    This function expects train_func to return (encoder, classifier),
    val_metrics and test_func to return a numpy array with five values:
    leaf accuracy, leaf precision, global accuracy, global precision,
    leaf auprc.
    """
    model_name = config['model_name']
    dataset_name = config['dataset_name']
    session_time = datetime.datetime.now().isoformat()
    checkpoint_dir = './weights/{}/{}'.format(model_name, dataset_name)
    if save_weights:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    if save_weights:
        val_metrics = model.fit(
            train_loader,
            val_loader,
            path='{}/{}.pt'.format(checkpoint_dir, session_time),
            best_path='{}/{}_best.pt'.format(checkpoint_dir, session_time),
            balanced=balanced
        )
    else:
        val_metrics = model.fit(
            train_loader,
            val_loader,
            balanced=balanced
        )
    test_output = model.test(test_loader)
    # test_metrics = metrics_func(
    #     test_output,
    #     display='both',
    #     compute_auprc=True
    # )
    # Report
    get_leaf_report(test_output, display='both')
    # Graph performance over epochs
    x = np.arange(config['epoch']) + 1
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(x, val_metrics[0], label='leaf accuracy')
    ax.plot(x, val_metrics[1], label='leaf precision')
    ax.plot(x, val_metrics[2], label='average accuracy')
    ax.plot(x, val_metrics[3], label='average precision')
    ax.set_xlabel('epoch')  # Add an x-label to the axes.
    ax.set_ylabel('score')  # Add a y-label to the axes.
    ax.set_title(
        "{} validation accuracy/precision over epochs".format(
            config['model_name']
        )
    )
    ax.legend()  # Add a legend.
    ax.grid()
    plt.savefig('{}/{}_perf.pdf'.format(checkpoint_dir, session_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='Pass a comma-separated list of PREPROCESSED dataset names (excluding .parquet) to use. This is a required argument.')
    parser.add_argument('-b', '--balanced', action='store_true', help='If specified, the system will rebalance class populations using weights. This may help with very imbalanced sets.')
    parser.add_argument('-n', '--dry_run', action='store_true', help='Don\'t save trained weights. Results are still logged to the logfile. Useful for when you run low on disk space.')
    parser.add_argument('-D', '--distilbert', action='store_true', help='If this flag is specified, download DistilBERT pretrained weights from huggingface to your user temp directory. By default, this repository tries to look for an offline-cached version instead.')
    parser.add_argument('-m', '--model', help="""Pass a comma-separated list of model names to run. Available models:
\tdb_bhcn\t\t(DistilBERT Branching Hierarchical Classifier)
\tdb_bhcn_awx\t\t(DistilBERT Branching Hierarchical Classifier + Adjacency Wrapping Matrix)
\tdb_ahmcnf\t\t(Adapted HMCN-F model running on DistilBERT encodings)
\tdb_achmcnn\t\t(Adapted C-HMCNN model running on DistilBERT encodings)
\ttfidf_hsgd\t\t(Internal-node SGD classifier hierarchy using tf-idf encodings)
By default, all models are run.""")
    parser.add_argument('-e', '--epoch', const=5, nargs='?', help='How many epochs to train DistilBERT models over. Default is 5.')
    parser.add_argument('-l', '--log', help='Path to log file. Default path is ./run.log.')
    parser.add_argument('-p', '--partial', action='store_true', help='Only run on 5% of each dataset (fixed seed). Useful for quick debugging runs.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print more information to the console (for debugging purposes).')
    parser.add_argument('-c', '--cpu', action='store_true', help='Only run on CPU. Use this if you have to run without CUDA support (warning: depressingly slow).')

    args = parser.parse_args()

    with open('./hyperparams.json', 'r') as j:
        hyperparams = json.loads(j.read())

    # Defaults
    verbose = False
    balanced = False
    log_path = './run.log'
    model_lst = [
        'db_bhcn',
        'db_bhcn_awx',
        'db_ahmcnf',
        'db_achmcnn',
        'tfidf_hsgd'
    ]
    epoch = 5
    save_weights = True
    full_set = True

    dataset_lst = [name.strip() for name in args.dataset.split(",")]
    if args.model:
        model_lst = [name.strip() for name in args.model.split(",")]
    if args.log:
        log_path = args.log
    if not args.distilbert:
        os.environ['TRANNSFORMERS_OFFLINE'] = '1'
    if args.epoch:
        epoch = int(args.epoch)
    if args.dry_run:
        save_weights = False
    if args.partial:
        full_set = False
    if args.balanced:
        balanced = True
    if args.verbose:
        verbose = args.verbose
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print('Using', device)

    distilbert.init()

    logging.basicConfig(filename=log_path, level=logging.INFO)

    if verbose:
        print()

    # Train models and log test set performance
    if 'db_bhcn' in model_lst:
        print('Training DB-BHCN...')
        logging.info('Training DB-BHCN...')
        config = hyperparams['db_bhcn']
        config['epoch'] = epoch
        config['device'] = device
        for dataset_name in dataset_lst:
            config['dataset_name'] = dataset_name
            print('Running on {}...'.format(dataset_name))
            logging.info('Running on {}...'.format(dataset_name))
            train_loader, val_loader, test_loader, hierarchy = dataset.get_loaders(
                './datasets/{}.parquet'.format(dataset_name),
                config,
                full_set=full_set,
                verbose=verbose,
                balanced=balanced
            )
            model = db_bhcn.DB_BHCN(hierarchy, config).to(device)
            train_val_test(
                config,
                model,
                train_loader,
                val_loader,
                test_loader,
                save_weights=save_weights,
                verbose=verbose,
                balanced=balanced,
                report=True
            )

    if 'db_bhcn_awx' in model_lst:
        print('Training DB-BHCN+AWX...')
        logging.info('Training DB-BHCN+AWX...')
        config = hyperparams['db_bhcn_awx']
        config['epoch'] = epoch
        config['device'] = device
        for dataset_name in dataset_lst:
            config['dataset_name'] = dataset_name
            print('Running on {}...'.format(dataset_name))
            logging.info('Running on {}...'.format(dataset_name))
            train_loader, val_loader, test_loader, hierarchy = dataset.get_loaders(
                './datasets/{}.parquet'.format(dataset_name),
                config,
                full_set=full_set,
                binary=True,
                build_R=True,
                verbose=verbose,
                balanced=balanced
            )
            model = db_bhcn.DB_BHCN(hierarchy, config, awx=True).to(device)
            train_val_test(
                config,
                model,
                train_loader,
                val_loader,
                test_loader,
                save_weights=save_weights,
                verbose=verbose,
                balanced=balanced,
                report=True
            )

    if 'db_ahmcnf' in model_lst:
        print('Training DB -> adapted HMCN-F...')
        logging.info('Training DB -> adapted HMCN-F...')
        config = hyperparams['db_ahmcnf']
        config['epoch'] = epoch
        config['device'] = device
        for dataset_name in dataset_lst:
            config['dataset_name'] = dataset_name
            print('Running on {}...'.format(dataset_name))
            logging.info('Running on {}...'.format(dataset_name))
            train_loader, val_loader, test_loader, hierarchy = dataset.get_loaders(
                './datasets/{}.parquet'.format(dataset_name),
                config,
                full_set=full_set,
                binary=True,
                verbose=verbose,
            )
            model = db_ahmcnf.DB_AHMCN_F(hierarchy, config).to(device)
            train_val_test(
                config,
                model,
                train_loader,
                val_loader,
                test_loader,
                save_weights=save_weights,
                verbose=verbose
            )

    if 'db_achmcnn' in model_lst:
        print('Training DB -> adapted C-HMCNN...')
        logging.info('Training DB -> adapted C-HMCNN...')
        config = hyperparams['db_achmcnn']
        config['epoch'] = epoch
        config['device'] = device
        for dataset_name in dataset_lst:
            config['dataset_name'] = dataset_name
            print('Running on {}...'.format(dataset_name))
            logging.info('Running on {}...'.format(dataset_name))
            train_loader, val_loader, test_loader, hierarchy = dataset.get_loaders(
                './datasets/{}.parquet'.format(dataset_name),
                config,
                full_set=full_set,
                binary=True,
                build_M=True,
                verbose=verbose,
            )
            model = db_achmcnn.DB_AC_HMCNN(hierarchy, config).to(device)
            train_val_test(
                config,
                model,
                train_loader,
                val_loader,
                test_loader,
                save_weights=save_weights,
                verbose=verbose
            )

    if 'tfidf_hsgd' in model_lst:
        print('Training tf-idf -> internal-node SGD classifier network...')
        logging.info('Training tf-idf -> internal-node SGD classifier network...')
        for dataset_name in dataset_lst:
            config = {}
            config['model_name'] = 'tfidf_hsgd'
            config['dataset_name'] = dataset_name
            print('Running on {}...'.format(dataset_name))
            logging.info('Running on {}...'.format(dataset_name))
            train_loader, test_loader, hierarchy = tfidf_hsgd.get_loaders(
                './datasets/{}.parquet'.format(dataset_name),
                config,
                full_set=full_set,
                verbose=verbose,
            )
            model = tfidf_hsgd.Tfidf_HSGD(config)
            train_val_test(
                config,
                model,
                train_loader,
                None,  # No validation set for pure ML
                test_loader,
                metrics_func=tfidf_hsgd.get_metrics,
                save_weights=save_weights,
                verbose=verbose
            )
