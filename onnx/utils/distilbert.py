"""Utilities for initialising and exporting DistilBERT instances."""
import os
import shutil
import transformers as tr
import bentoml

base_encoder = None
base_encoder_state = None
tokenizer = None

def get_pretrained():
    """Return a DistilBERT instance with distilbert-base-uncased loaded."""
    global base_encoder
    global base_encoder_state
    if base_encoder is None or base_encoder_state is None:
        base_encoder = tr.DistilBertModel.from_pretrained(
            'distilbert-base-uncased'
        )
        base_encoder_state = base_encoder.state_dict()
    encoder = base_encoder
    encoder.load_state_dict(base_encoder_state)
    return encoder

def get_tokenizer():
    """Return a DistilBERTFastTokenizer instance."""
    global tokenizer
    if tokenizer is None:
        tokenizer = tr.DistilBertTokenizerFast.from_pretrained(
            'distilbert-base-uncased'
        )
    return tokenizer

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
    # transformers.onnx
    model.eval()
    name = '{}_{}'.format(classifier_name, dataset_name)
    if bento:
        bentoml.transformers.save(
            'encoder_' + name,
            model=model,
            tokenizer=get_tokenizer()
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
        shutil.rmtree('tmp/')


if __name__ == "__main__":
    pass
