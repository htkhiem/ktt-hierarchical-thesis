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

from utils.metric import get_metrics

from models import db_bhcn, db_ahmcnf, db_achmcnn, db_linear, model_sklearn, tfidf_hsgd, tfidf_lsgd

from utils import dataset, distilbert


def repeat_train(
        config,
        model,
        train_loader,
        val_loader,
        test_loader,
        repeat,
        metrics_func=get_metrics,
        save_weights=True,
        verbose=False
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
    all_test_metrics = np.zeros((repeat, 5), dtype=float)
    checkpoint_dir = './weights/{}/{}'.format(model_name, dataset_name)
    if save_weights:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    for i in range(repeat):
        print('RUN #{} ----'.format(i))
        logging.info('RUN #{} ----'.format(i))
        if save_weights:
            model.fit(
                train_loader,
                val_loader,
                path='{}/run_{}.pt'.format(checkpoint_dir, repeat),
                best_path='{}/run_{}_best.pt'.format(checkpoint_dir, repeat)
            )
        else:
            model.fit(
                train_loader,
                val_loader,
            )
        test_output = model.test(test_loader)
        test_metrics = metrics_func(
            test_output,
            display='both',
            compute_auprc=True
        )
        all_test_metrics[i, :] = test_metrics
    averaged = np.average(all_test_metrics, axis=0)
    averaged_display = '--- Average of {} runs:\nLeaf accuracy: {}\nLeaf precision: {}\nPath accuracy: {}\nPath precision: {}\nLeaf AU(PRC): {}'.format(
        repeat, averaged[0], averaged[1], averaged[2], averaged[3], averaged[4])
    print(averaged_display)
    logging.info(averaged_display)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='Pass a comma-separated list of PREPROCESSED dataset names (excluding .parquet) to use. This is a required argument.')
    parser.add_argument('-n', '--dry_run', action='store_true', help='Don\'t save trained weights. Results are still logged to the logfile. Useful for when you run low on disk space.')
    parser.add_argument('-D', '--distilbert', action='store_true', help='If this flag is specified, download DistilBERT pretrained weights from huggingface to your user temp directory. By default, this repository tries to look for an offline-cached version instead.')
    parser.add_argument('-m', '--model', help="""Pass a comma-separated list of model names to run. Available models:
\tdb_bhcn\t\t(DistilBERT Branching Hierarchical Classifier)
\tdb_bhcn_awx\t\t(DistilBERT Branching Hierarchical Classifier + Adjacency Wrapping Matrix)
\tdb_ahmcnf\t\t(Adapted HMCN-F model running on DistilBERT encodings)
\tdb_achmcnn\t\t(Adapted C-HMCNN model running on DistilBERT encodings)
\tdb_linear\t\t(DistilBERT+Linear layer)
\ttfidf_hsgd\t\t(Internal-node SGD classifier hierarchy using tf-idf encodings)
\ttfidf_lsgd\t\t(Leaf node SGD classifier hierarchy using tf-idf encodings)
By default, all models are run.""")
    parser.add_argument('-e', '--epoch', const=5, nargs='?', help='How many epochs to train DistilBERT models over. Default is 5.')
    parser.add_argument('-R', '--run', const=5, nargs='?', help='How many times to repeat training. The final result will be an average of all the runs. Default is 5.')
    parser.add_argument('-l', '--log', help='Path to log file. Default path is ./run.log.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print more information to the console (for debugging purposes).')
    parser.add_argument('-c', '--cpu', action='store_true', help='Only run on CPU. Use this if you have to run without CUDA support (warning: depressingly slow).')

    args = parser.parse_args()

    with open('./hyperparams.json', 'r') as j:
        hyperparams = json.loads(j.read())

    # Defaults
    verbose = False
    log_path = './run.log'
    model_lst = [
        'db_bhcn',
        'db_bhcn_awx',
        'db_ahmcnf',
        'db_achmcnn',
        'tfidf_hsgd',
        'tfidf_lsgd'
    ]
    epoch = 5
    repeat = 5
    save_weights = True

    dataset_lst = [name.strip() for name in args.dataset.split(",")]
    if args.model:
        model_lst = [name.strip() for name in args.model.split(",")]
    if args.log:
        log_path = args.log
    if not args.distilbert:
        os.environ['TRANNSFORMERS_OFFLINE'] = '1'
    if args.epoch:
        epoch = int(args.epoch)
    if args.run:
        repeat = int(args.run)
    if args.dry_run:
        save_weights = False
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
        print('Testing DB-BHCN...')
        logging.info('Testing DB-BHCN...')
        config = hyperparams['db_bhcn']
        config['epoch'] = epoch
        config['device'] = device
        for dataset_name in dataset_lst:
            config['dataset_name'] = dataset_name
            print('Running on {}...'.format(dataset_name))
            logging.info('Running on {}...'.format(dataset_name))
            train_loader, val_loader, test_loader, hierarchy = dataset.get_loaders(
                dataset_name,
                config,
                verbose=verbose,
            )
            model = db_bhcn.DB_BHCN(hierarchy, config).to(device)
            repeat_train(
                config,
                model,
                train_loader,
                val_loader,
                test_loader,
                repeat,
                save_weights=save_weights,
                verbose=verbose
            )

    if 'db_bhcn_awx' in model_lst:
        print('Testing DB-BHCN+AWX...')
        logging.info('Testing DB-BHCN+AWX...')
        config = hyperparams['db_bhcn_awx']
        config['epoch'] = epoch
        config['device'] = device
        for dataset_name in dataset_lst:
            config['dataset_name'] = dataset_name
            print('Running on {}...'.format(dataset_name))
            logging.info('Running on {}...'.format(dataset_name))
            train_loader, val_loader, test_loader, hierarchy = dataset.get_loaders(
                dataset_name,
                config,
                verbose=verbose,
            )
            model = db_bhcn.DB_BHCN(hierarchy, config, awx=True).to(device)
            repeat_train(
                config,
                model,
                train_loader,
                val_loader,
                test_loader,
                repeat,
                save_weights=save_weights,
                verbose=verbose
            )

    if 'db_ahmcnf' in model_lst:
        print('Testing DB -> adapted HMCN-F...')
        logging.info('Testing DB -> adapted HMCN-F...')
        config = hyperparams['db_ahmcnf']
        config['epoch'] = epoch
        config['device'] = device
        for dataset_name in dataset_lst:
            config['dataset_name'] = dataset_name
            print('Running on {}...'.format(dataset_name))
            logging.info('Running on {}...'.format(dataset_name))
            train_loader, val_loader, test_loader, hierarchy = dataset.get_loaders(
                dataset_name,
                config,
                verbose=verbose,
            )
            model = db_ahmcnf.DB_AHMCN_F(hierarchy, config).to(device)
            repeat_train(
                config,
                model,
                train_loader,
                val_loader,
                test_loader,
                repeat,
                save_weights=save_weights,
                verbose=verbose
            )

    if 'db_achmcnn' in model_lst:
        print('Testing DB -> adapted C-HMCNN...')
        logging.info('Testing DB -> adapted C-HMCNN...')
        config = hyperparams['db_achmcnn']
        config['epoch'] = epoch
        config['device'] = device
        for dataset_name in dataset_lst:
            config['dataset_name'] = dataset_name
            print('Running on {}...'.format(dataset_name))
            logging.info('Running on {}...'.format(dataset_name))
            train_loader, val_loader, test_loader, hierarchy = dataset.get_loaders(
                dataset_name,
                config,
                verbose=verbose,
            )
            model = db_achmcnn.DB_AC_HMCNN(hierarchy, config).to(device)
            repeat_train(
                config,
                model,
                train_loader,
                val_loader,
                test_loader,
                repeat,
                save_weights=save_weights,
                verbose=verbose
            )

    if 'db_linear' in model_lst:
        print('Training DB -> Linear...')
        logging.info('Training DB -> Linear...')
        config = hyperparams['db_linear']
        config['epoch'] = epoch
        config['device'] = device
        for dataset_name in dataset_lst:
            config['dataset_name'] = dataset_name
            print('Running on {}...'.format(dataset_name))
            logging.info('Running on {}...'.format(dataset_name))
            train_loader, val_loader, test_loader, hierarchy = dataset.get_loaders(
                dataset_name,
                config,
                verbose=verbose,
            )
            model = db_linear.DB_Linear(hierarchy, config).to(device)
            repeat_train(
                config,
                model,
                train_loader,
                val_loader,
                test_loader,
                repeat,
                metrics_func=db_linear.get_metrics,
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
            train_loader, test_loader, hierarchy = model_sklearn.get_loaders(
                dataset_name,
                config,
                verbose=verbose,
            )
            model = tfidf_hsgd.Tfidf_HSGD(config)
            repeat_train(
                config,
                model,
                train_loader,
                None,  # No validation set for pure ML
                test_loader,
                repeat,
                metrics_func=model_sklearn.get_metrics,
                save_weights=save_weights,
                verbose=verbose
            )

    if 'tfidf_lsgd' in model_lst:
        print('Training tf-idf -> leaf-node SGD classifier network...')
        logging.info('Testing tf-idf -> leaf-node SGD classifier network...')
        for dataset_name in dataset_lst:
            config = {}
            config['model_name'] = 'tfidf_lsgd'
            config['dataset_name'] = dataset_name
            print('Running on {}...'.format(dataset_name))
            logging.info('Running on {}...'.format(dataset_name))
            train_loader, test_loader, _ = model_sklearn.get_loaders(
                dataset_name,
                config,
                verbose=verbose,
            )
            model = tfidf_lsgd.Tfidf_LSGD(config)
            repeat_train(
                config,
                model,
                train_loader,
                None,  # No validation set for pure ML
                test_loader,
                repeat,
                metrics_func=model_sklearn.get_metrics,
                save_weights=save_weights,
                verbose=verbose
            )
