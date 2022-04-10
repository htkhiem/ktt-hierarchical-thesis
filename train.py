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

from models import PYTORCH_MODEL_LIST
from models import SKLEARN_MODEL_LIST

from utils import cli

# Don't touch this
MODEL_LIST = PYTORCH_MODEL_LIST + SKLEARN_MODEL_LIST


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
        ref_df.to_parquet('{}/last_{}_reference.parquet'.format(checkpoint_dir, start_datetime))


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
    Pass a comma-separated list of model names to run. Available models:
    db_bhcn\t\t(DistilBERT Branching Hierarchical Classifier)
    db_bhcn_awx\t\t(DistilBERT Branching Hierarchical Classifier + Adjacency Wrapping matriX)
    db_ahmcnf\t\t(Adapted HMCN-F model running on DistilBERT encodings)
    db_achmcnn\t\t(Adapted C-HMCNN model running on DistilBERT encodings)
    db_linear\t\t(DistilBERT+Linear layer)
    tfidf_hsgd\t\t(Internal-node SGD classifier hierarchy using tf-idf encodings)
    tfidf_lsgd\t\t(Leaf node SGD classifier hierarchy using tf-idf encodings)
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

    def init_config(model_name, display_name):
        """Announce training and load configuration with common parameters."""
        click.echo(
            '{}Training {}...{}'.format(cli.BOLD, display_name, cli.PLAIN)
        )
        logging.info('Training {}...'.format(display_name))
        try:
            config = hyperparams[model_name]
            config['epoch'] = epoch
            config['device'] = device
            return config
        except KeyError:
            # sklearn models do not have configuration.
            return {
                'model_name': model_name
            }

    def init_dataset(dataset_name, loader_func, config):
        config['dataset_name'] = dataset_name
        click.echo(
            '{}Runnning on {}...{}'.format(cli.CYAN, dataset_name, cli.PLAIN)
        )
        logging.info('Running on {}...'.format(dataset_name))
        train_loader, val_loader, test_loader, hierarchy = loader_func(
            dataset_name,
            config,
        )
        return (
            train_loader,
            val_loader,
            test_loader,
            hierarchy,
            config
        )
    # Defaults
    model_lst = MODEL_LIST[:]
    dataset_lst = [name.strip() for name in datasets.split(",")]
    if len(models) > 0:
        model_lst = [name.strip() for name in models.split(",")]
    os.environ['TRANNSFORMERS_OFFLINE'] = '1' if not distilbert else '0'
    device = 'cuda' if torch.cuda.is_available() and not cpu else 'cpu'
    print('Using', device.upper())

    # Check if PyTorch models are requested
    if any(m in model_lst for m in PYTORCH_MODEL_LIST):
        model_pytorch = __import__(
            'models', globals(), locals(), ['model_pytorch'], 0).model_pytorch

    # Check if Scikit-learn models are requested
    if any(m in model_lst for m in SKLEARN_MODEL_LIST):
        model_sklearn = __import__(
            'models', globals(), locals(), ['model_sklearn'], 0).model_sklearn

    logging.basicConfig(filename=log_path, level=logging.INFO)

    # Train models and log test set performance
    if 'db_bhcn' in model_lst:
        config = init_config('db_bhcn', 'DB-BHCN')
        DB_BHCN = __import__(
            'models', globals(), locals(), [], 0).DB_BHCN
        for dataset_name in dataset_lst:
            (
                train_loader, val_loader, test_loader, hierarchy, config
            ) = init_dataset(
                dataset_name, model_pytorch.get_loaders, config
            )
            model = DB_BHCN(hierarchy, config).to(device)
            train_and_test(
                config,
                model,
                train_loader,
                val_loader,
                test_loader,
                metrics_func=model_pytorch.get_metrics,
                dry_run=dry_run,
                verbose=verbose,
                gen_reference=reference,
                dvc=dvc
            )

    if 'db_bhcn_awx' in model_lst:
        config = init_config('db_bhcn_awx', 'DB-BHCN+AWX')
        DB_BHCN_AWX = __import__(
            'models', globals(), locals(), [], 0).DB_BHCN_AWX
        for dataset_name in dataset_lst:
            (
                train_loader, val_loader, test_loader, hierarchy, config
            ) = init_dataset(
                dataset_name, model_pytorch.get_loaders, config
            )
            model = DB_BHCN_AWX(hierarchy, config).to(device)
            train_and_test(
                config,
                model,
                train_loader,
                val_loader,
                test_loader,
                metrics_func=model_pytorch.get_metrics,
                dry_run=dry_run,
                verbose=verbose,
                gen_reference=reference,
                dvc=dvc
            )

    if 'db_ahmcnf' in model_lst:
        config = init_config('db_ahmcnf', 'DistilBERT + Adapted HMCN-F')
        DB_AHMCN_F = __import__(
            'models', globals(), locals(), [], 0).DB_AHMCN_F
        for dataset_name in dataset_lst:
            (
                train_loader, val_loader, test_loader, hierarchy, config
            ) = init_dataset(
                dataset_name, model_pytorch.get_loaders, config
            )
            model = DB_AHMCN_F(hierarchy, config).to(device)
            train_and_test(
                config,
                model,
                train_loader,
                val_loader,
                test_loader,
                metrics_func=model_pytorch.get_metrics,
                dry_run=dry_run,
                verbose=verbose,
                dvc=dvc
            )

    if 'db_achmcnn' in model_lst:
        config = init_config('db_achmcnn', 'DistilBERT + Adapted C-HMCNN')
        DB_AC_HMCNN = __import__(
            'models', globals(), locals(), [], 0).DB_AC_HMCNN
        for dataset_name in dataset_lst:
            (
                train_loader, val_loader, test_loader, hierarchy, config
            ) = init_dataset(
                dataset_name, model_pytorch.get_loaders, config
            )
            model = DB_AC_HMCNN(hierarchy, config).to(device)
            train_and_test(
                config,
                model,
                train_loader,
                val_loader,
                test_loader,
                metrics_func=model_pytorch.get_metrics,
                dry_run=dry_run,
                verbose=verbose,
                dvc=dvc
            )

    if 'db_linear' in model_lst:
        config = init_config('db_linear', 'DistilBERT+Linear')
        DB_Linear = __import__(
            'models', globals(), locals(), [], 0).DB_Linear
        db_linear_metrics = __import__(
            'models', globals(), locals(), [], 0).get_linear_metrics
        for dataset_name in dataset_lst:
            (
                train_loader, val_loader, test_loader, hierarchy, config
            ) = init_dataset(
                dataset_name, model_pytorch.get_loaders, config
            )
            model = DB_Linear(hierarchy, config).to(device)
            train_and_test(
                config,
                model,
                train_loader,
                val_loader,
                test_loader,
                metrics_func=db_linear_metrics,
                dry_run=dry_run,
                verbose=verbose,
                dvc=dvc
            )

    if 'tfidf_hsgd' in model_lst:
        config = init_config('tfidf_hsgd', 'Tf-idf + Hierarchical SGD')
        Tfidf_HSGD = __import__(
            'models', globals(), locals(), [], 0).Tfidf_HSGD
        for dataset_name in dataset_lst:
            (
                train_loader, val_loader, test_loader, hierarchy, config
            ) = init_dataset(
                dataset_name, model_sklearn.get_loaders, config
            )
            model = Tfidf_HSGD(config)
            train_and_test(
                config,
                model,
                train_loader,
                None,  # No validation set for pure ML
                test_loader,
                metrics_func=model_sklearn.get_metrics,
                dry_run=dry_run,
                verbose=verbose,
                dvc=dvc
            )

    if 'tfidf_lsgd' in model_lst:
        config = init_config('tfidf_lsgd', 'Tf-idf + Leaf SGD')
        Tfidf_LSGD = __import__(
            'models', globals(), locals(), [], 0).Tfidf_LSGD
        for dataset_name in dataset_lst:
            (
                train_loader, val_loader, test_loader, hierarchy, config
            ) = init_dataset(
                dataset_name, model_sklearn.get_loaders, config
            )
            model = Tfidf_LSGD(config)
            train_and_test(
                config,
                model,
                train_loader,
                None,  # No validation set for pure ML
                test_loader,
                metrics_func=model_sklearn.get_metrics,
                dry_run=dry_run,
                verbose=verbose,
                dvc=dvc
            )


if __name__ == "__main__":
    main()