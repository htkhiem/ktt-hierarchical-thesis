"""Utilities for initialising and exporting DistilBERT instances."""
import transformers as tr
import os
import bentoml

def get_pretrained():
    """Return a DistilBERT instance with distilbert-base-uncased loaded."""
    encoder = base_encoder
    encoder.load_state_dict(base_encoder_state)
    return encoder


def export_trained(
        model,
        dataset_name,
        classifier_name,
        bento=False
):
    """
    Export a fine-tuned instance of DistilBERT.

    If BentoML support is specified, then ONNX exporting is skipped and a model
    is saved to the default Bento model store directly.
    Note: init() needs to be called at least once before this function can
    be used.
    """
    # Load model with pretrained weights.
    # We also need to export a pretrained tokenizer along to babysit
    # transformers.onnx.2
    model.eval()
    name = '{}_{}'.format(classifier_name, dataset_name)
    if bento:
        bentoml.transformers.save(
            'encoder_' + name,
            model=model,
            tokenizer=tokenizer
        )
    else:
        # Export into transformers model .bin format
        tmp_path = 'tmp/distilbert_' + name
        model.save_pretrained(tmp_path)
        tokenizer.save_pretrained(tmp_path)
        # Run PyTorch ONNX exporter on said model file
        onnx_path = 'output/{}/encoder'.format(name)
        os.system(
            'python3 -m transformers.onnx --model {} {} --opset 11'.format(
                tmp_path, onnx_path
            )
        )
        os.rmdir('tmp/')


def init():
    """Initialise the backup global DistilBERT instance."""
    global tokenizer
    global base_encoder
    global base_encoder_state
    tokenizer = tr.DistilBertTokenizerFast.from_pretrained(
        'distilbert-base-uncased'
    )
    base_encoder = tr.DistilBertModel.from_pretrained(
        'distilbert-base-uncased'
    )
    base_encoder_state = base_encoder.state_dict()


if __name__ == "__main__":
    pass
