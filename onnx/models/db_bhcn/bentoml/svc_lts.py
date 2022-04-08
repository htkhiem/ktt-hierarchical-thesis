"""Service file for DB-BHCN + Walmart_30k."""
import requests
from typing import List

import numpy as np
import pandas as pd
import torch

import bentoml
from bentoml.adapters import JsonInput
from bentoml.frameworks.transformers import TransformersModelArtifact
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.service.artifacts.common import JSONArtifact
from bentoml.types import JsonSerializable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([
    TransformersModelArtifact('encoder'),
    PytorchModelArtifact('classifier'),
    JSONArtifact('hierarchy'),
    JSONArtifact('config'),

])
class DB_BHCN(bentoml.BentoService):
    """Real-time inference service for DB-BHCN."""

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
        self.classes = hierarchy['classes']
        # Load service configuration JSON
        self.monitoring_enabled = self.artifacts.config['monitoring_enabled']
        # We use PyTorch-based Transformers
        self.encoder.to(device)

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
        # Classify using our classifier head
        scores = self.classifier(encoder_cls).cpu().detach().numpy()
        # Segmented argmax, as usual
        pred_codes = np.array([
            np.argmax(
                scores[
                    :,
                    self.level_offsets[level]:
                    self.level_offsets[level + 1]
                ],
                axis=1
            ) + self.level_offsets[level]
            for level in range(len(self.level_sizes))
        ], dtype=int)

        predicted_names = np.array(['\n'.join([self.classes[level] for level in row]) for row in pred_codes.swapaxes(1, 0)])

        if self.monitoring_enabled:
            new_rows = {
                'targets': predicted_names[:, -1].tolist()
            }
            # Shape as generated is (microbatch, feature/class), but we need to
            # iterate over features/classes to generate each column of the
            # appending set. To do that, we transpose the matrices. This does
            # not copy any data and is thus performant.
            for col_idx, embs in enumerate(encoder_cls.swapaxes(1, 0)):
                new_rows[str(col_idx)] = embs.astype(np.float64)
            for col_idx, scores in enumerate(scores.swapaxes(1, 0)):
                new_rows[self.classes[col_idx]] = scores.astype(np.float64)

            new_row_df = pd.DataFrame(new_rows)
            requests.post(
                "http://localhost:5000/iterate",
                data=new_row_df.to_json(),
                headers={"content-type": "application/json"},
            )
        return predicted_names
