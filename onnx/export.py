"""
Model export controller.

This script controls the exporting of models, either directly to ONNX,
or to a BentoML ModelStorage.
"""
import click
from textwrap import dedent
import json
import glob
import torch
import pandas as pd

from models.model_list import PYTORCH_MODEL_LIST
from models.model_list import SKLEARN_MODEL_LIST

from utils import cli

# Don't touch this
MODEL_LIST = PYTORCH_MODEL_LIST + SKLEARN_MODEL_LIST


def get_path(
        model_name, dataset_name, best=False, time=None, reference_set=False
):
    """
    Grabs latest checkpoint in the automatic folder structure.

    Parameters
    ----------
    best: bool
        If true, get paths for 'best' weights instead of last-epoch.
    time: str
        An ISO datetime string (no microseconds). If time is specified, get
        that iteration specifically. If not, use the last-run session.
    reference_set: bool
        If true, return the name of the reference dataset of that
        training session instead of the checkpoint. best` is not
        applicable in this case (as the reference set is only generated)
        for the last epoch).

    Returns
    -------
    path: str
        Either the path to the corresponding model checkpoint, or to the reference
        dataset.
    """
    if reference_set:
        if best:
            click.echo(
                '{}WARNING:{} get_path(best=True) ignored as it has '
                'been requested to return the path '
                'to the reference set instead.'.format(cli.RED, cli.PLAIN)
            )
        set_names = sorted(glob.glob(
            'weights/{}/{}/last_{}_reference.parquet'.format(
                model_name,
                dataset_name,
                time if time is not None else '*'
            )
        ))
        if len(set_names) < 1:
            click.echo(
                '{}WARNING:{} No matching reference dataset found.'.format(
                    cli.RED, cli.PLAIN)
            )
            return None
        print('Using reference dataset at', set_names[-1])
        return set_names[-1]
    weight_names = sorted(glob.glob(
        'weights/{}/{}/{}_{}.pt'.format(
            model_name,
            dataset_name,
            'best' if best else 'last',
            time if time is not None else '*'
        )
    ))
    return weight_names[-1]


@click.command()
@click.argument('datasets', required=1)
@click.option(
    '-m', '--models', default='', show_default=False,
    help=dedent("""Pass a comma-separated list of model names to run. Available models:
    db_bhcn\t\t(DistilBERT Branching Hierarchical Classifier)
    db_bhcn_awx\t\t(DistilBERT Branching Hierarchical Classifier + Adjacency Wrapping Matrix)
    db_ahmcnf\t\t(Adapted HMCN-F model running on DistilBERT encodings)
    db_achmcnn\t\t(Adapted C-HMCNN model running on DistilBERT encodings)
    db_linear\t\t(DistilBERT + Linear layer)
    tfidf_hsgd\t\t(Internal-node SGD classifier hierarchy using tf-idf encodings)
    tfidf_lsgd\t\t(ILeaf node SGD classifier hierarchy using tf-idf encodings)
    By default, all models are exported. An error will be raised if a model has
    not been trained with any of the specified datasets.""")
)
@click.option(
    '-t', '--time', help=dedent("""
    Optionally specify which trained weights to load by their time.
    """)
)
@click.option(
    '--best/--latest', default=True, show_default=True,
    help='User best-epoch weights instead of latest-epoch.'
)
@click.option(
    '-b', '--bento/--no-bento', default=True, show_default=True,
    help='Add exported models to the local BentoML model store.'
)
@click.option(
    '-v', '--verbose', is_flag=True, default=False,
    help='Print more information to the console (for debugging purposes).'
)
@click.option(
    '-c', '--cpu/--gpu', is_flag=True, default=False,
    help=dedent("""
    Only trace models using CPU. Use this if you have to run without CUDA
    support.
    """)
)
def main(
        datasets,
        models,
        time,
        best,
        bento,
        verbose,
        cpu
):
    """Dispatch export operations."""
    print('Using', 'best' if best else 'latest')
    # Defaults
    with open('./hyperparams.json', 'r') as j:
        hyperparams = json.loads(j.read())
    model_lst = MODEL_LIST
    dataset_lst = [name.strip() for name in datasets.split(",")]
    if len(models) > 0:
        model_lst = [name.strip() for name in models.split(",")]
    device = 'cuda' if torch.cuda.is_available() and not cpu else 'cpu'
    print('Using', device.upper())

    for dataset_name in dataset_lst:
        if 'db_bhcn' in model_lst:
            click.echo('{}Exporting {}...{}'.format(
                cli.BOLD, 'DB-BHCN', cli.PLAIN))
            db_bhcn = __import__(
                'models', globals(), locals(), ['db_bhcn'], 0).db_bhcn
            model = db_bhcn.DB_BHCN.from_checkpoint(
                get_path('db_bhcn', dataset_name, best=best, time=time),
            ).to(device)
            reference_set = pd.read_parquet(get_path(
                'db_bhcn', dataset_name, time=time, reference_set=True))
            model.export(dataset_name, bento, reference_set)

        if 'db_bhcn_awx' in model_lst:
            click.echo('{}Exporting {}...{}'.format(
                cli.BOLD, 'DB-BHCN+AWX', cli.PLAIN))
            db_bhcn = __import__(
                'models', globals(), locals(), ['db_bhcn'], 0).db_bhcn
            model = db_bhcn.DB_BHCN.from_checkpoint(
                get_path('db_bhcn_awx', dataset_name, best, time),
            ).to(device)
            reference_set = pd.read_parquet(get_path(
                'db_bhcn_awx', dataset_name, time=time, reference_set=True))
            model.export(dataset_name, bento, reference_set)

        if 'db_ahmcnf' in model_lst:
            click.echo('{}Exporting {}...{}'.format(
                cli.BOLD, 'DistilBERT+Adapted HMCN-F', cli.PLAIN))
            db_ahmcnf = __import__(
                'models', globals(), locals(), ['db_ahmcnf'], 0).db_ahmcnf
            model = db_ahmcnf.DB_AHMCN_F.from_checkpoint(
                get_path('db_ahmcnf', dataset_name, best, time)
            ).to(device)
            model.export(dataset_name, bento)

        if 'db_achmcnn' in model_lst:
            click.echo('{}Exporting {}...{}'.format(
                cli.BOLD, 'DistilBERT+Adapted C-HMCNN', cli.PLAIN))
            db_achmcnn = __import__(
                'models', globals(), locals(), ['db_achmcnn'], 0).db_achmcnn
            model = db_achmcnn.DB_AC_HMCNN.from_checkpoint(
                get_path('db_achmcnn', dataset_name, best, time)
            ).to(device)
            model.export(dataset_name, bento)

        if 'db_linear' in model_lst:
            click.echo('{}Exporting {}...{}'.format(
                cli.BOLD, 'DistilBERT+Linear', cli.PLAIN))
            db_linear = __import__(
                'models', globals(), locals(), ['db_linear'], 0).db_linear
            model = db_linear.DB_Linear.from_checkpoint(
                get_path('db_linear', dataset_name, best, time)
            ).to(device)
            model.export(dataset_name, bento)

        if 'tfidf_hsgd' in model_lst:
            click.echo('{}Exporting {}...{}'.format(
                cli.BOLD, 'Tf-idf + Hierarchical SGD', cli.PLAIN))
            tfidf_hsgd = __import__(
                'models', globals(), locals(), ['tfidf_hsgd'], 0).tfidf_hsgd
            model = tfidf_hsgd.Tfidf_HSGD.from_checkpoint(
                get_path('tfidf_hsgd', dataset_name, best, time)
            )
            model.export(dataset_name, bento)

        if 'tfidf_lsgd' in model_lst:
            click.echo('{}Exporting {}...{}'.format(
                cli.BOLD, 'Tf-idf + Leaf SGD', cli.PLAIN))
            tfidf_lsgd = __import__(
                'models', globals(), locals(), ['tfidf_lsgd'], 0).tfidf_lsgd
            model = tfidf_lsgd.Tfidf_LSGD.from_checkpoint(
                get_path('tfidf_lsgd', dataset_name, best, time)
            )
            model.export(dataset_name, bento)


if __name__ == "__main__":
    main()
