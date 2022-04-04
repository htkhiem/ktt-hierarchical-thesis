"""Build script for generating DB-BHCN Bentos."""
import os
import pandas as pd
import shutil
import click
from textwrap import dedent

import bentoml
from mako.template import Template
import yaml

@click.command()
@click.argument('dataset')
@click.option('-e', '--encoder-id', default='latest', show_default=True, help=dedent("""
Optionally specify which BentoML encoder model version ID to use.
"""))
@click.option('-c', '--classifier-id', default='latest', show_default=True, help=dedent("""
Optionally specify which BentoML classifier model version ID to use.
"""))
@click.option(
    '-m', '--monitoring/--no-monitoring', is_flag=True, default=True,
    help=dedent("""Enable/disable Evidently model performance monitoring. This
    will be skipped if the chosen BentoML was not bundled with a reference
    dataset.
"""))
def build_service(dataset, encoder_id, classifier_id, monitoring):
    """Build a DB-BHCN inference service.

    Please specify the dataset with which your desired instance has been
    trained against and optionally a BentoML version ID. By default, the
    latest found matching BentoML model is used.
    """
    svc_template = Template(filename='db_bhcn.template.py')
    encoder_name = 'encoder_db_bhcn_{}:{}'.format(dataset.lower(), encoder_id)
    classifier_name = 'classifier_db_bhcn_{}:{}'.format(
        dataset.lower(), classifier_id)
    click.echo('Using encoder: {}\nUsing classifier: {}'.format(
        encoder_name, classifier_name
    ))

    # Create a same-directory copy of the monitoring code.
    # This makes it possible for BentoML to pre-run the service for
    # checking as in the BentoService, these two files will be in the same directory.
    shutil.copy('../monitoring.py', '_monitoring.py')

    monitoring_available = False
    model = bentoml.models.get(classifier_name)
    hierarchy = model.info.metadata['hierarchy']
    if monitoring:
        if 'reference' in model.info.metadata.keys():
            monitoring_available = True
            pd.DataFrame(model.info.metadata['reference']).to_parquet(
                '_references.parquet'
            )
            with open("evidently.template.yaml", 'r') as evidently_template:
                config = yaml.safe_load(evidently_template)
            config['prediction'] = hierarchy['classes'][
                hierarchy['level_offsets'][-2]:hierarchy['level_offsets']
                [-1]
            ]
            with open("_evidently.yaml", 'w') as evidently_config_file:
                yaml.dump(config, evidently_config_file)
        else:
            click.echo(dedent("""WARNING: No bundled reference dataset
            found in BentoML model. Evidently metrics will be disabled.
            Please ensure you have exported the model with the -r flag.
            """))

    with open('_svc.py', 'w') as svcfile:
        svcfile.write(svc_template.render(
            encoder=encoder_name,
            classifier=classifier_name,
            dataset=dataset.lower(),
            monitoring='True' if monitoring_available else 'False'
        ))
    file_list = ['_svc.py']
    python_packages = [
        'numpy',
        'pandas',
        'tensorflow',
        'nvidia-pyindex',
        'nvidia-cufft',
        'nvidia-curand',
        'nvidia-cublas',
        'nvidia-cudnn',
        'onnxruntime-gpu',
        'onnx',
        'torch',
        'transformers',
    ]
    if monitoring_available:
        file_list += [
            '_monitoring.py',
            '_references.parquet',
            '_evidently.yaml',
            '_reference.parquet'
        ]
        python_packages += [
            'flask',
            'evidently'
        ]

    bentoml.build(
        '_svc.py:svc',
        labels={
            'owner': 'ktt',
            'stage': 'demo'
        },
        description="file: ./README.md",
        include=file_list,
        exclude=['build_service.py'],
        docker={
            'gpu': True
        },
        python={
            'packages': python_packages
        }
    )

    # Clear temporary files
    os.remove('_svc.py')
    os.remove('_monitoring.py')
    os.remove('_evidently.yaml')
    os.remove('_references.parquet')


if __name__ == '__main__':
    build_service()
