"""
This file defines an abstract class for model definition.

Every model implementation in this framework is a subclass of this.
This file also serves as documentation for future reference.
"""
from abc import ABC, abstractmethod


class Model(ABC):
    """Abstract model class.

    A model class contains the model implementation, checkpoint load-save
    methods, training and testing routines as well as facilities for
    exporting a trained instance to ONNX/Bento.

    It should also contain a PyTorch-like forward method for single-row/batch
    forward propagation. This method needs not be public but should be used
    within the training and testing methods for consistency.
    """

    @classmethod
    @abstractmethod
    def from_checkpoint(cls, path):
        """
        Construct model from saved checkpoint.

        If the model's topology depends on the specific dataset it was trained
        against, this will only work with checkpoints that were created with
        the config file saved within.
        """
        pass

    @abstractmethod
    def fit(
            self,
            train_loader,
            val_loader,
            path=None,
            best_path=None,
            resume_from=None
    ):
        """Training script."""
        pass

    @abstractmethod
    def save(self, path, optim=None):
        """
        Save current instance state to disk.

        It also takes in optimiser state for resuming capability.
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Load a saved state from disk to the current instance.

        The current instance must be properly configured (i.e. model topology
        must match the instance that created said saved state).
        It returns optimiser state as extracted from the saved state, which is
        useful for resuming the optimiser in the training script.
        """
        pass

    @abstractmethod
    def export(self, dataset_name, bento=False):
        """
        Export current instance to ONNX or Bento.

        Internally, the Bento option must also use ONNX graphs (through ONNX
        Runtime). Transformer models however should only export their
        classifiers as ONNX, while the transformers themselves should
        be exported using BentoML's internal transformers exporting tool.
        """
        pass
