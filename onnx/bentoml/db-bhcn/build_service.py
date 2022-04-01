"""Build script for generating DB-BHCN Bentos."""
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import shutil
import click
from textwrap import dedent

import torch
import onnxruntime
import bentoml
from mako.template import Template


def generate_reference_data(dataset, encoder_name, classifier_name, cpu=False):
    """Load the model and run on test set to capture reference data."""
    def chunkify(df, size):
        """Split a Dataframe into chunks.

        It is a minimal equivalent of a non-randomised DataLoader.
        """
        return (df.iloc[pos:pos + size] for pos in range(0, len(df), size))

    device = 0 if torch.cuda.is_available() and not cpu else -1
    if device == -1:
        print('Using CPU')
    print('Using CUDA')

    test_set = pd.read_parquet('../../datasets/{}/test.parquet'.format(dataset))

    # PyTorch-based
    _, encoder, tokenizer = bentoml.transformers.load(
        encoder_name,  # also packs tokeniser
        # tasks='feature-extraction',  # barebones pipeline
        # device=device,
    )
    encoder.to(device)
    encoder.eval()
    session_options = onnxruntime.SessionOptions()
    session_options.add_session_config_entry('execution_mode', 'sequential')
    classifier_runner = bentoml.onnx.load_runner(
        classifier_name,
        # Automatic fallback to CPU if GPU isn't available.
        # This need not be explicitly specified.
        backend="onnxruntime-gpu",
        # TensorRT provider is bugging out even with the correct version.
        providers=['CUDAExecutionProvider'],
        session_options=session_options,
    )

    hierarchy = bentoml.models.get(
        classifier_name
    ).info.metadata

    leaf_size = hierarchy['level_sizes'][-1]
    leaf_start = hierarchy['level_offsets'][-2]
    leaf_end = hierarchy['level_offsets'][-1]

    features = np.empty((0, 768), dtype=np.float32)
    predictions = np.empty((0, leaf_size))
    targets = test_set['codes'].apply(lambda lst: hierarchy['classes'][
            leaf_start + lst[-1]
        ]).to_numpy()

    print('Generating reference dataset...')

    with torch.no_grad():
        for chunk in tqdm(chunkify(test_set, 64)):
            inputs = tokenizer(
                chunk.name.to_list(),
                None,
                add_special_tokens=True,  # CLS, SEP
                max_length=64,
                padding='max_length',
                truncation=True
            )
            encoder_outputs = encoder(
                torch.tensor(
                    inputs['input_ids'], dtype=torch.long, device=device
                ),
                torch.tensor(
                    inputs['attention_mask'], dtype=torch.long, device=device
                )
            )
            last_hidden_layer = encoder_outputs['last_hidden_state'][:, 0, :].cpu()
            features = np.concatenate(
                [features, last_hidden_layer.cpu()], axis=0
            )
            classifier_outputs = classifier_runner.run_batch(
                last_hidden_layer)[0]
            # print(classifier_outputs.shape)
            predictions = np.concatenate(
                [
                    predictions,
                    classifier_outputs[:, leaf_start:leaf_end]
                ],
                axis=0
            )

    reference_col_dict = {
        'targets': targets
    }
    for col_idx in range(768):
        reference_col_dict[str(col_idx)] = features[:, col_idx]
    for col_idx in range(predictions.shape[1]):
        reference_col_dict[
            hierarchy['classes'][leaf_start + col_idx]
        ] = predictions[:, col_idx]
    reference_set = pd.DataFrame(reference_col_dict)
    reference_set.to_parquet('./reference.parquet')


@click.command()
@click.argument('dataset')
@click.option('-e', '--encoder-id', default='latest', show_default=True, help=dedent("""
Optionally specify which BentoML encoder model version ID to use.
"""))
@click.option('-c', '--classifier-id', default='latest', show_default=True, help=dedent("""
Optionally specify which BentoML classifier model version ID to use.
"""))
@click.option('--cpu/--gpu', is_flag=True, default=False, help=dedent("""
Only use CPU to generate the reference dataset even if a GPU is available.

If you do not have a GPU, you do not need to specify this as the exporter
will automatically fall back to CPU mode.
"""))
def build_service(dataset, encoder_id, classifier_id, cpu):
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
    with open('_svc.py', 'w') as svcfile:
        svcfile.write(svc_template.render(
            encoder=encoder_name,
            classifier=classifier_name
        ))
    # Create a same-directory copy of the monitoring code.
    # This makes it possible for BentoML to pre-run the service for
    # checking as in the BentoService, these two files will be in the same directory.
    shutil.copy('../monitoring.py', 'monitoring.py')

    # Generate reference data
    generate_reference_data(dataset, encoder_name, classifier_name, cpu)

    bentoml.build(
        '_svc.py:svc',
        labels={
            'owner': 'ktt',
            'stage': 'demo'
        },
        description="file: ./README.md",
        include=['_svc.py', 'monitoring.py', 'references.parquet', 'evidently.yaml'],
        exclude=['build_service.py'],
        docker={
            'gpu': True
        },
        python={
            'packages': [
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
                'evidently'
            ]
        }
    )

    # Clear temporary files
    os.remove('_svc.py')
    os.remove('monitoring.py')


if __name__ == '__main__':
    build_service()
