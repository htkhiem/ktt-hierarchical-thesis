"""
Hyperparameter tuning controller.

Use this Python script to search for the best hyperparameters for your own
datasets using Ray Tune.
"""
import os
import click
from textwrap import dedent

import logging
import json
import torch

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.trial import Trial
from ray.tune.schedulers import ASHAScheduler

import models as mds

from utils import cli


class TrialTerminationReporter(CLIReporter):
    def __init__(self):
        super().__init__()
        self.num_terminated = 0

    def should_report(self, trials, done=False):
        """Reports only on trial termination events."""
        old_num_terminated = self.num_terminated
        self.num_terminated = len([t for t in trials if t.status == Trial.TERMINATED])
        return self.num_terminated > old_num_terminated


def tune_model(
        tune_config,
        model_class,
        train_loader,
        val_loader,
        test_loader,
        hierarchy,
        metrics_func,
        verbose=False
):
    """Tune a model's hyperparameters over a dataset.

    This function expects a special configuration file, named 't
    une.py'.
    This file contains ranges and modes (uniform, choice, or fixed) for each of
    the hyperparameters.

    For now, the metric of choice is leaf-level accuracy.
    """

    def trial(config):
        model = model_class(hierarchy, config).to(config['device'])
        val_metrics = model.fit(
            train_loader,
            val_loader,
        )
        last_epoch_leaf_acc = val_metrics[0, -1]
        tune.report(leaf_acc=last_epoch_leaf_acc)

    search_space = {
        'model_name': tune_config['model_name'],
        'epoch': tune_config['epoch'],
        'device': tune_config['device']
    }
    # Construct Ray Tune search space from config
    for hyperparam in tune_config['mode'].keys():
        mode = tune_config['mode'][hyperparam]
        if mode == 'fixed':
            search_space[hyperparam] = tune_config['range'][hyperparam]
        elif mode == 'uniform':
            search_space[hyperparam] = tune.uniform(
                tune_config['range'][hyperparam][0],
                tune_config['range'][hyperparam][1],
            )
        elif mode == 'choice':
            search_space[hyperparam] = tune.choice(
                tune_config['range'][hyperparam]
            )
        else:
            raise NotImplementedError(
                '{} mode is not currently supported'.format(mode)
            )
    # Init Ray object store
    ray.init(object_store_memory=tune_config['object_store']*10**9)
    use_cuda = tune_config['device'] == 'cuda'
    analysis = tune.run(
        trial,
        config=search_space,
        metric='leaf_acc',
        mode='max',
        num_samples=tune_config['sample'],
        scheduler=ASHAScheduler(metric='leaf_acc', mode="max"),
        resources_per_trial={
            'cpu': 1 if not use_cuda else 0,
            'gpu': 1 if use_cuda else 0
        },
        progress_reporter=TrialTerminationReporter()
    )

    return analysis


@click.command()
@click.argument('datasets', required=1)
@click.option(
    '-D', '--distilbert', is_flag=True, default=False, show_default=True,
    help=dedent("""
    Specify this flag to download DistilBERT pretrained weights from
    huggingface to your user temp directory. By default, we will try
    to look for an offline-cached version instead.
    """)
)
@click.option(
    '-m', '--models', default='', show_default=False, help=dedent("""
    Pass a comma-separated list of model names to run.
    By default, all models are run.
    """)
)
@click.option(
    '-e', '--epoch', default=5, show_default=True, help=dedent("""
    How many epochs to train DistilBERT models over.
    scikit-learn models are trained until they converge or a maximum of
    1000 iterations is reached.
    """)
)
@click.option(
    '-s', '--sample', default=5, show_default=True, help=dedent("""
    Number of Ray Tune samples.
    """)
)
@click.option(
    '--object-store', default=4, show_default=True, help=dedent("""
    Maximum Ray object store size in GB. Adjust this if you have more than 4GB
    at your disposal for storing multiple trials at once, or if you run into
    memory issues.
    """)
)
@click.option(
    '-l', '--log-path', default='./run.log', show_default=True, help=dedent("""
    Path to log file. Default path is ./run.log.
    """)
)
@click.option(
    '-v', '--verbose', is_flag=True, default=False, show_default=True,
    help=dedent("""
    Print more information to the console (for debugging purposes).
    """)
)
@click.option(
    '-c', '--cpu', is_flag=True, default=False, show_default=True,
    help=dedent("""
    Only run on CPU. Use this if you have to run without CUDA support (warning:
    depressingly slow).
    """)
)
def main(
        datasets,
        distilbert,
        models,
        epoch,
        sample,
        object_store,
        log_path,
        verbose,
        cpu,
):
    """Dispatch training and testing jobs."""
    with open('./tune.json', 'r') as j:
        hyperparams = json.loads(j.read())

    def init_config(model_name):
        """Announce training and load configuration with common parameters."""
        try:
            config = hyperparams[model_name]
            logging.info('Training {}...'.format(config['display_name']))
            click.echo(
                '{}Training {}...{}'.format(
                    cli.BOLD, config['display_name'], cli.PLAIN
                )
            )
            config['model_name'] = model_name
            if 'display_name' not in config.keys():
                config['display_name'] = model_name
            config['epoch'] = epoch
            config['device'] = device
            config['sample'] = sample
            config['object_store'] = object_store
            return config
        except KeyError:
            # Model does not have any configurable hyperparameter
            logging.info('Tuning {}...'.format(model_name))
            click.echo(
                '{}Tuning {}...{}'.format(cli.BOLD, model_name, cli.PLAIN)
            )
            return {
                'model_name': model_name,
                'display_name': model_name,
                'sample': sample,
                'object_store': object_store
            }

    def init_dataset(dataset_name, loader_func, preprocessor, config):
        config['dataset_name'] = dataset_name
        click.echo(
            '{}Runnning on {}...{}'.format(cli.CYAN, dataset_name, cli.PLAIN)
        )
        logging.info('Running on {}...'.format(dataset_name))
        train_loader, val_loader, test_loader, hierarchy = loader_func(
            dataset_name,
            tune_config,
            preprocessor,
            shuffle=False,  # Keep everything consistent for tuning
            verbose=verbose
        )
        return (
            train_loader,
            val_loader,
            test_loader,
            hierarchy,
            config
        )
    # Defaults
    model_lst = mds.MODELS.keys()
    dataset_lst = [name.strip() for name in datasets.split(",")]
    if len(models) > 0:
        model_lst = [name.strip() for name in models.split(",")]
    os.environ['TRANSFORMERS_OFFLINE'] = '1' if not distilbert else '0'
    device = 'cuda' if torch.cuda.is_available() and not cpu else 'cpu'
    print('Using', device.upper())

    logging.basicConfig(filename=log_path, level=logging.INFO)

    # Train models and log test set performance
    for model_name in model_lst:
        tune_config = init_config(model_name)
        model_class = mds.MODELS[model_name]
        for dataset_name in dataset_lst:
            (
                train_loader, val_loader, test_loader, hierarchy, config
            ) = init_dataset(
                dataset_name,
                model_class.get_dataloader_func(),
                model_class.get_preprocessor(tune_config),
                tune_config
            )
            analysis = tune_model(
                tune_config,
                model_class,
                train_loader,
                val_loader,
                test_loader,
                hierarchy,
                metrics_func=model_class.get_metrics_func(),
                verbose=verbose,
            )
            print(analysis.best_config)
            logging.info(analysis.best_config)

            # Plot by epoch
            ax = None  # This plots everything on the same plot
            for d in analysis.trial_dataframes.values():
                ax = d.leaf_acc.plot(ax=ax, legend=False)


if __name__ == "__main__":
    main()
