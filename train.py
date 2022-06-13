"""
Training controller.

Use this Python script to initiate batched training of any combination of model
and dataset.
"""
import os
import click
from textwrap import dedent
import datetime
import logging
import json
import torch

import models as mds

from utils import cli


def train_and_test(
        config,
        model,
        train_loader,
        val_loader,
        test_loader,
        metrics_func,
        gen_reference=False,
        dvc=True,
        dry_run=False,
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
    checkpoint_dir = './weights/{}/{}'.format(model_name, dataset_name)
    start_datetime = datetime.datetime.now().replace(microsecond=0).isoformat()
    if not dry_run:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        model.fit(
            train_loader,
            val_loader,
            path='{}/last_{}.pt'.format(checkpoint_dir, start_datetime),
            best_path='{}/best_{}.pt'.format(checkpoint_dir, start_datetime),
            dvc=dvc
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
    report = dedent("""
    Test set metrics:
    \tLeaf accuracy:  {}
    \tLeaf precision: {}
    \tPath accuracy:  {}
    \tPath precision: {}
    \tLeaf AU(PRC):   {}
    """.format(
        test_metrics[0],
        test_metrics[1],
        test_metrics[2],
        test_metrics[3],
        test_metrics[4],
    ))
    print(report)
    logging.info(report)

    if gen_reference:
        ref_df = model.gen_reference_set(test_loader)
        ref_df.to_parquet('{}/reference_{}.parquet'.format(checkpoint_dir, start_datetime))


@click.command()
@click.argument('datasets', required=1)
@click.option(
    '-n', '--dry-run', is_flag=True, default=False, show_default=True,
    help=dedent("""
    Do not save trained weights. Results are still logged to the logfile.
    Useful for when you run low on disk space.
    """)
)
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
    '-l', '--log-path', default='./run.log', show_default=True, help=dedent("""
    Path to log file. Default path is ./run.log.
    """)
)
@click.option(
    '-r', '--reference', is_flag=True, default=False, help=dedent("""
    Generate an Evidently-compatible reference set. Specify this flag if
    you wish to monitor the model in production using Evidently.
    """)
)
@click.option(
    '--dvc/--no-dvc', '--verbose', default=True, show_default=True,
    help=dedent("""
    Track model weights using DVC.
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
        dry_run,
        distilbert,
        models,
        epoch,
        reference,
        dvc,
        log_path,
        verbose,
        cpu,
):
    """Dispatch training and testing jobs."""
    with open('./hyperparams.json', 'r') as j:
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
            return config
        except KeyError:
            # Model does not have any configurable hyperparameter
            logging.info('Training {}...'.format(model_name))
            click.echo(
                '{}Training {}...{}'.format(cli.BOLD, model_name, cli.PLAIN)
            )
            return {
                'model_name': model_name,
                'display_name': model_name,
            }

    def init_dataset(dataset_name, loader_func, preprocessor, config):
        config['dataset_name'] = dataset_name
        click.echo(
            '{}Runnning on {}...{}'.format(cli.CYAN, dataset_name, cli.PLAIN)
        )
        logging.info('Running on {}...'.format(dataset_name))
        train_loader, val_loader, test_loader, hierarchy = loader_func(
            dataset_name,
            config,
            preprocessor,
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

    click.echo('{}Training these models:{}'.format(cli.BOLD, cli.PLAIN))
    for model_name in model_lst:
        print('\t' + model_name)
    click.echo('{}on these datasets:{}'.format(cli.BOLD, cli.PLAIN))
    for dataset_name in dataset_lst:
        print('\t' + dataset_name)

    # Train models and log test set performance
    for model_name in model_lst:
        config = init_config(model_name)
        model_class = mds.MODELS[model_name]
        for dataset_name in dataset_lst:
            (
                train_loader, val_loader, test_loader, hierarchy, config
            ) = init_dataset(
                dataset_name,
                model_class.get_dataloader_func(),
                model_class.get_preprocessor(config),
                config
            )
            model = model_class(hierarchy, config).to(device)
            train_and_test(
                config,
                model,
                train_loader,
                val_loader,
                test_loader,
                metrics_func=model_class.get_metrics_func(),
                dry_run=dry_run,
                verbose=verbose,
                gen_reference=reference,
                dvc=dvc
            )

    click.echo('\n{}Finished!{}'.format(cli.GREEN, cli.PLAIN))


if __name__ == "__main__":
    main()
