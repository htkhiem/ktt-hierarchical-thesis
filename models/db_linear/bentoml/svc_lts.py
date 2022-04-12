"""Service file for DB-Linear + Walmart_30k."""
import os
import requests
from typing import List
import json

import numpy as np
import torch

import bentoml
from bentoml.adapters import JsonInput
from bentoml.frameworks.transformers import TransformersModelArtifact
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.service.artifacts.common import JSONArtifact
from bentoml.types import JsonSerializable

EVIDENTLY_HOST = os.environ.get('EVIDENTLY_HOST', 'localhost')
EVIDENTLY_PORT = os.environ.get('EVIDENTLY_PORT', 5001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REFERENCE_SET_FEATURE_POOL = 32
POOLED_FEATURE_SIZE = 768 // REFERENCE_SET_FEATURE_POOL

@bentoml.env(
    requirements_txt_file='models/db_linear/bentoml/requirements.txt',
    docker_base_image='bentoml/model-server:0.13.1-py36-gpu'
)
@bentoml.artifacts([
    TransformersModelArtifact('encoder'),
    PytorchModelArtifact('classifier'),
    JSONArtifact('hierarchy'),
    JSONArtifact('config'),
])
class DB_Linear(bentoml.BentoService):
    """Real-time inference service for DB-Linear."""

    _initialised = False

    def init_fields(self):
        """Initialise the necessary fields. This is not a constructor."""
        self.tokeniser = self.artifacts.encoder.get('tokenizer')
        self.encoder = self.artifacts.encoder.get('model')
        self.classifier = self.artifacts.classifier
        # Load hierarchical metadata
        hierarchy = self.artifacts.hierarchy
        self.level_sizes = hierarchy['level_sizes']
        self.level_offsets = hierarchy['level_offsets']
        self.leaf_offsets = hiearchy['level_offsets'][-2]
        self.classes = hierarchy['classes']
        # Load service configuration JSON
        self.monitoring_enabled = self.artifacts.config['monitoring_enabled']
        # We use PyTorch-based Transformers
        self.encoder.to(device)
        # Identical pool layer as in the test script.
        self.pool = torch.nn.AvgPool1d(REFERENCE_SET_FEATURE_POOL)

        self._initialised = True

    @bentoml.api(
        input=JsonInput(),
        batch=True,
        mb_max_batch_size=64,
        mb_max_latency=2000,
    )
    def predict(self, parsed_json_list: List[JsonSerializable]):
        """Classify text to the trained hierarchy."""
        if not self._initialised:
            self.init_fields()
        texts = [j['text'] for j in parsed_json_list]
        # Pre-processing: tokenisation
        tokenised = self.tokeniser(
            texts,
            None,
            add_special_tokens=True,  # CLS, SEP
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
            # DistilBERT tokenisers return attention masks by default
        )
        # Encode using DistilBERT
        encoder_cls = self.encoder(
            tokenised['input_ids'].to(device),
            tokenised['attention_mask'].to(device)
        )[0][:, 0, :]
        encoder_cls_pooled = self.pool(encoder_cls)
        # Classify using our classifier head
        scores = self.classifier(encoder_cls).cpu().detach().numpy()
        # Segmented argmax, as usual
        pred_codes = np.array([
            np.argmax(
                scores
                ,
                axis=1
            ) + self.leaf_offsets
        ], dtype=int)

        predicted_names = np.array([
            [self.classes[level] for level in row]
            for row in pred_codes.swapaxes(1, 0)
        ])

        if self.monitoring_enabled:
            """
            Create a 2D list contains the following content:
            [:, 0]: leaf target names (left as zeroes)
            [:, 1:25]: pooled features,
            [:, 25:]: leaf classification scores.
            The first axis is the microbatch axis.
            """
            new_rows = np.zeros(
                (len(texts), 1 + POOLED_FEATURE_SIZE + self.level_sizes[-1]),
                dtype=np.float64
            )
            new_rows[
                :,
                1:POOLED_FEATURE_SIZE+1
            ] = encoder_cls_pooled.cpu().detach().numpy()
            new_rows[
                :,
                POOLED_FEATURE_SIZE+1:
            ] = scores[:, self.level_offsets[-2]:]
            requests.post(
                "http://{}:{}/iterate".format(EVIDENTLY_HOST, EVIDENTLY_PORT),
                data=json.dumps({'data': new_rows.tolist()}),
                headers={"content-type": "application/json"},
            )
        return ['\n'.join(row) for row in predicted_names]
