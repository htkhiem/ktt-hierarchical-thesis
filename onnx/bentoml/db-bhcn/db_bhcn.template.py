"""
Service file for DB-BHCN + Walmart_30k.
"""
import requests
import json
import bentoml
import torch
import pandas as pd
from bentoml.io import Text
import numpy as np
import onnxruntime


# Workaround for a weird implementation quirk inside transformers'
# pipelines/base.py that compare passed device with 0:
#  self.device = device if framework == "tf" else
#                    torch.device("cpu" if device < 0 else
#                    f"cuda:{device}")
device = 0 if torch.cuda.is_available() else -1

# Use load_runner for optimised performance
# The transformer encoder section uses a directly-exported model, instead of
# through ONNX, for optimisation.


# PyTorch-based
encoder_runner = bentoml.transformers.load_runner(
    '${encoder}',  # also packs tokeniser
    tasks='feature-extraction',  # barebones pipeline
    device=device
)

session_options = onnxruntime.SessionOptions()
session_options.add_session_config_entry('execution_mode', 'sequential')
classifier_runner = bentoml.onnx.load_runner(
    '${classifier}',
    # Automatic fallback to CPU if GPU isn't available.
    # This need not be explicitly specified.
    # backend="onnxruntime-gpu",
    # TensorRT provider is bugging out even with the correct version.
    providers=['CUDAExecutionProvider'],
    session_options=session_options
)

hierarchy = bentoml.models.get(
    '${classifier}'
).info.metadata

leaf_size = hierarchy['level_sizes'][-1]
leaf_start = hierarchy['level_offsets'][-2]
leaf_end = hierarchy['level_offsets'][-1]

level_offsets = hierarchy['level_offsets']
level_sizes = hierarchy['level_sizes']
classes = hierarchy['classes']

svc = bentoml.Service(
    'db_bhcn_${dataset}',
    runners=[encoder_runner, classifier_runner]
)


@svc.api(input=Text(), output=Text())
def predict(input_text: str) -> np.ndarray:
    """Define the entire inference process for this model."""
    # Pre-processing: tokenisation
    encoder_outputs = encoder_runner.run(input_text)
    last_hidden_layer = np.array(
        encoder_outputs, dtype=np.float32)[:, 0, :][0]
    classifier_outputs = classifier_runner.run(last_hidden_layer)

    scores = classifier_outputs[0]

    # Segmented argmax, as usual
    pred_codes = [
        int(
            np.argmax(
                scores[
                    level_offsets[level]:level_offsets[level + 1]
                ]
            ) + level_offsets[level]
        )
        for level in range(len(level_sizes))
    ]

    new_row = {
        'targets': [classes[pred_codes[-1]]]
    }
    for col_idx, emb in enumerate(last_hidden_layer):
        new_row[str(col_idx)] = [emb.astype(np.float64)]
    for col_idx, score in enumerate(scores):
        new_row[classes[col_idx]] = [score.astype(np.float64)]

    new_row_df = pd.DataFrame(new_row)
    requests.post(
        "http://localhost:5000/iterate",
        data=new_row_df.to_json(),
        headers={"content-type": "application/json"},
    )

    return '\n'.join([classes[i] for i in pred_codes])
