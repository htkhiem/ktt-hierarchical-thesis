"""Utilities for initialising and exporting DistilBERT instances."""
import os
import shutil
import transformers as tr
import bentoml
import tempfile
import torch

from .encoder import BasePreprocessor


def get_pretrained():
    """Return a DistilBERT instance with distilbert-base-uncased loaded."""
    return tr.DistilBertModel.from_pretrained(
        'distilbert-base-uncased'
    )


def get_tokenizer():
    """Return a DistilBERTFastTokenizer instance."""
    return tr.DistilBertTokenizerFast.from_pretrained(
        'distilbert-base-uncased'
    )


def export_trained(
        model,
        path,
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
    # Export into transformers model .bin format temporarily
    with tempfile.TemporaryDirectory() as tempdir:
        model.save_pretrained(tempdir.name)
        get_tokenizer().save_pretrained(tempdir.name)
        # Run PyTorch ONNX exporter on said model file
        os.system(
            'python3 -m transformers.onnx --model {} {} --opset 11'.format(
                tempdir.name, path
            )
        )


class DistilBertPreprocessor(BasePreprocessor):
    """Preprocessor (tokenisation) class for DistilBERT.

    KTT provides basic facilities for several Transformer encoders.
    """

    def __init__(self, config):
        """Construct a preprocessor instance.

        All instances currently use the same DistilBERT tokeniser instance.
        """
        super().__init__(config)
        self.tokeniser = get_tokenizer()
        self.max_len = config['max_len']

    def __call__(self, text):
        """Tokenise incoming text into DistilBERT IDs and mask.

        Parameters
        ----------
        text: str
            The text to tokenise.

        Returns
        -------
        inputs: dict
            A dictionary containing the following fields:

            - ``ids``: DistilBERT token IDs
            - ``mask``: DistilBERT LM mask (all 1s)
        """
        tokenised = self.tokeniser(
            text,
            None,  # No text_pair
            add_special_tokens=True,  # CLS, SEP
            max_length=self.max_len,
            padding='max_length',
            truncation=True
            # BERT tokenisers return attention masks by default
        )
        return {
            'ids': torch.tensor(tokenised['input_ids'], dtype=torch.long),
            'mask': torch.tensor(tokenised['attention_mask'], dtype=torch.long)
        }


if __name__ == "__main__":
    pass
