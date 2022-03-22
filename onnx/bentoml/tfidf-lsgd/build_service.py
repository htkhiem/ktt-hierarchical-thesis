"""Build script for generating TFIDF-LSGD Bentos."""
import argparse
import bentoml
from mako.template import Template
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--classifier', required=True,
        help='Bento model name of the classifier.'
    )
    args = parser.parse_args()

    svc_template = Template(filename='tfidf_lsgd.template.py')
    with open('_svc.py', 'w') as svcfile:
        svcfile.write(svc_template.render(
            classifier=args.classifier
        ))

    bentoml.build(
        '_svc.py:svc',
        labels={
            'owner': 'ktt',
            'stage': 'demo'
        },
        description="file: ./README.md",
        include=['_svc.py'],
        exclude=['build_service.py'],
        docker={
            # The very reason why we can't use YAML: broken parsing for
            # DockerOptions right where it matters.
            # This is needed to get a base image with CUDA Runtime
            # preinstalled.
            'gpu': False
        },
        python={
            'packages': [
                'numpy',
                'pandas',
                'tensorflow',
                'onnxruntime-gpu',
                'onnx',
                'sklearn',
                'nltk'
            ]
        }
    )

    os.remove('_svc.py')
