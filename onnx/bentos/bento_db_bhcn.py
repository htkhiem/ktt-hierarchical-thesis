import bentoml
from bentoml.io import Text, NumpyNdarray
import numpy as np

# Use load_runner for optimised performance
# The transformer encoder section uses a directly-exported model, instead of
# through ONNX, for optimisation.
encoder_runner = bentoml.transformers.load_runner(
    'encoder_db_bhcn_walmart_30k:latest',  # also packs tokeniser
    tasks='feature-extraction'  # barebones pipeline
)
classifier_runner = bentoml.onnx.load_runner(
    'classifier_db_bhcn_walmart_30k:latest'
)
hierarchy = bentoml.models.get(
    'classifier_db_bhcn_walmart_30k:latest'
).info.metadata

level_offsets = hierarchy['level_offsets']
level_sizes = hierarchy['level_sizes']
classes = hierarchy['classes']

svc = bentoml.Service(
    'db_bhcn_walmart_30k',
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

    # Segmented argmax, as usual
    pred_codes = [
        int(
            np.argmax(
                scores[
                    level_offsets[level]:level_offsets[level + 1]]
            ) + level_offsets[level]
        )
        for level in range(len(level_sizes))
    ]
    return '\n'.join([classes[i] for i in pred_codes])
