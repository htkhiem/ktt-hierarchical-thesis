"""
Service file for DB-Linear + Walmart_30k.
"""

import bentoml
import torch
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

leaf_offset = hierarchy['level_offsets'][-2]
level_sizes = hierarchy['level_sizes']
classes = hierarchy['classes']


svc = bentoml.Service(
    'db_linear',
    runners=[encoder_runner, classifier_runner]
)


@svc.api(input=Text(), output=Text())
def predict(input_text: str) -> np.ndarray:
    """Define the entire inference process for this model."""
    # Pre-processing: tokenisation
    encoder_outputs = encoder_runner.run(input_text)
    last_hidden_layer = np.array(encoder_outputs)[:, 0, :][0]
    classifier_outputs = classifier_runner.run(last_hidden_layer)

    scores = classifier_outputs[0]

    pred_code = int(
        np.argmax(scores) + leaf_offset
    )
    return classes[pred_code]
