"""
Model export controller.

This script controls the exporting of models, either directly to ONNX,
or to a BentoML ModelStorage.
"""
import os
import sys
import shutil
import yaml
import click
from textwrap import dedent
import json
import glob
import torch

import models as mds
from utils.build import init_folder_structure
from utils import cli


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
        Either the path to the corresponding model checkpoint, or to the
        reference dataset.
    """
    if reference_set:
        if best:
            click.echo(
                '{}WARNING:{} get_path(best=True) ignored as it has '
                'been requested to return the path '
                'to the reference set instead.'.format(cli.RED, cli.PLAIN)
            )
        set_names = sorted(glob.glob(
            'weights/{}/{}/reference_{}.parquet'.format(
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
    pathname = 'weights/{}/{}/{}_{}.*'.format(
        model_name,
        dataset_name,
        'best' if best else 'last',
        time if time is not None else '*'
    )
    weight_names = sorted(glob.glob(
        pathname
    ) + glob.glob(
        pathname + '.dvc'
    ))
    if len(weight_names) < 1:
        print('{}ERROR:{} No trained weights found for {} on {}!'.format(
            cli.RED,
            cli.PLAIN,
            model_name,
            dataset_name
        ))
        sys.exit(1)
    if weight_names[-1].endswith('.dvc'):
        name_no_dvc = weight_names[-1][:-4]
        if not os.path.exists(name_no_dvc):
            os.system('dvc checkout {}.dvc'.format(weight_names[-1]))
        return name_no_dvc
    return weight_names[-1]


def export_model(
        model,
        model_name,
        dataset_name,
        bento=False,
        reference_set_path=False
):
    """Export a given model.

    This function takes care of paths and configuration files for the exporting
    process. Models need only implement details specific to them.
    """
    name = '{}_{}'.format(
        model_name,
        dataset_name
    )
    if not bento:
        # Create path
        enc_path = 'output/{}/encoder/'.format(name)
        cls_path = 'output/{}/classifier/'.format(name)
        if os.path.exists(enc_path):
            shutil.rmtree(enc_path)
        os.makedirs(enc_path)
        if not os.path.exists(cls_path):
            shutil.rmtree(cls_path)
        os.makedirs(cls_path)
        model.export_onnx(cls_path, enc_path)
    else:
        # Export as BentoML service
        build_path = 'build/{}'.format(name)
        if os.path.exists(build_path):
            shutil.rmtree(build_path)
        build_path_inference = ''
        instance_config, svc = model.export_bento_resources(svc_config={
            'monitoring_enabled': reference_set_path is not None
        })
        if reference_set_path is not None:
            # If a path to a reference dataset is available, export the
            # model as a service with monitoring capabilities.
            with open(
                    'models/{}/bentoml/evidently.yaml'.format(model_name), 'r'
            ) as evidently_template:
                evidently_config = yaml.safe_load(evidently_template)
                evidently_config.update(instance_config)

            # Init folder structure, Evidently YAML and so on.
            build_path_inference = init_folder_structure(
                build_path,
                {
                    'reference_set_path': reference_set_path,
                    'grafana_dashboard_path':
                        'models/{}/bentoml/dashboard.json'.format(model_name),
                    'evidently_config': evidently_config
                }
            )
        else:
            # Init folder structure for a minimum system (no monitoring)
            build_path_inference = init_folder_structure(build_path)
        # Export the BentoService to the correct path.
        svc.save_to_dir(build_path_inference)


@click.command()
@click.argument('datasets', required=1)
@click.option(
    '-m', '--models', default='', show_default=False,
    help=dedent("""Pass a comma-separated list of model names to run.
    By default, all models are exported. An error will be raised if a model has
    not been trained with any of the specified datasets.""")
)
@click.option(
    '-t', '--time', help=dedent("""
    Optionally specify which trained weights to load by their time.
    """)
)
@click.option(
    '--best/--latest', default=False, show_default=True,
    help='User best-epoch weights instead of latest-epoch.'
)
@click.option(
    '-b', '--bento/--no-bento', default=True, show_default=True,
    help='Add exported models to the local BentoML model store.'
)
@click.option(
    '--monitoring/--no-monitoring', is_flag=True, default=True,
    show_default=True,
    help=dedent("""Enable/disable Evidently model performance monitoring. This
    will be skipped if the chosen BentoML was not bundled with a reference
    dataset.""")
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
        monitoring,
        cpu
):
    """Dispatch export operations."""
    print('Using', 'best' if best else 'latest')
    # Defaults
    with open('./hyperparams.json', 'r') as j:
        hyperparams = json.loads(j.read())
    model_lst = mds.MODELS.keys()
    dataset_lst = [name.strip() for name in datasets.split(",")]
    if len(models) > 0:
        model_lst = [name.strip() for name in models.split(",")]
    device = 'cuda' if torch.cuda.is_available() and not cpu else 'cpu'
    print('Using', device.upper())

    for dataset_name in dataset_lst:
        for model_name in model_lst:
            config = hyperparams[model_name]
            display_name = config['display_name'] if 'display_name' in\
                config.keys() else model_name
            click.echo('{}Exporting {}...{}'.format(
                cli.BOLD, display_name, cli.PLAIN))
            model = mds.MODELS[model_name].from_checkpoint(
                get_path(model_name, dataset_name, best=best, time=time),
            ).to(device)
            if monitoring:
                reference_set_path = get_path(
                    model_name, dataset_name, time=time, reference_set=True)
                if reference_set_path is not None:
                    export_model(model, model_name,
                                 dataset_name, bento, reference_set_path)
                    return
            export_model(model, model_name, dataset_name, bento)


if __name__ == "__main__":
    main()
