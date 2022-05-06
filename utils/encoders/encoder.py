"""Base encoder definitions."""

class BasePreprocessor:
    """A base class for your custom preprocessors.

    This base preprocessor does not do anything (passthrough). It's used
    by default when your model does not specify any preprocessor.

    Models may require specific preprocessing in the form of tokenisation,
    stemming, word removal and so on. Such preprocessing can be implemented
    by subclassing this class in your own model definition file.
    """

    def __init__(self, config):
        """General constructor."""
        self.config = config

    def __call__(self, text):
        """Transform the given text into your preferred input format.

        Parameters
        ----------
        text: str
            The text to tokenise.

        Returns
        -------
        inputs: dict
            All preprocessors must return a dictionary of model input fields.
            Do not use the key ``label`` as it is reserved for use by the
            ``PyTorchDataset`` class, which will later write the labels
            corresponding to this text to the dictionary.

            By default, the BaseProcessor returns the input unchanged as the
            ``text`` key.
        """
        return {'text': text}
