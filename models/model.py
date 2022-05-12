"""
This file defines an abstract class for model definition.

Every model implementation in this framework is a subclass of this.
This file also serves as documentation for future reference.
"""
from abc import ABC, abstractmethod
from utils.encoders.encoder import BasePreprocessor

class Model(ABC):
    """Abstract model class for all frameworks.

    A model class contains the model implementation, checkpoint load-save
    methods, training and testing routines as well as facilities for
    exporting a trained instance to ONNX/BentoML.

    It should also contain a PyTorch-like forward method for single-row/batch
    forward propagation. This method needs not be public but should be used
    within the training and testing methods for consistency.
    """

    @classmethod
    def get_preprocessor(cls, config):
        """Return an instance of this model's preferred preprocessor."""
        return BasePreprocessor(config)

    @classmethod
    def get_dataloader_func(cls):
        """Return the function that generates dataloaders for this model."""
        pass

    @classmethod
    def get_metrics_func(cls):
        """Return the metris function compatible with this model."""
        pass

    @classmethod
    @abstractmethod
    def from_checkpoint(cls, path):
        """
        Construct model from saved checkpoint.

        If the model's topology depends on the specific dataset it was trained
        against, this will only work with checkpoints that were created with
        the config file saved within.

        Parameters
        ----------
        path : str
            Path to the checkpoint.

        Returns
        -------
        instance : Model
            An instance of this model, initialised to the specified checkpoint.

        See also
        --------
        save : Create a checkpoint readable by this method.
        load : An alternative to this method for already-constructed instances.
        """
        pass

    @abstractmethod
    def fit(
            self,
            train_loader,
            val_loader,
            path=None,
            best_path=None,
            resume_from=None,
            dvc=True
    ):
        """Training script for this model.

        Parameters
        ----------
        train_loader : collections.abc.Iterable
            The training set packaged into any format suitable for
            this model, preferably something iterable. PyTorch models can use
            their Datasets or DataLoader objects.
        val_loader : collections.abc.Iterable
            The validation set, similarly packaged.
        path : str, optional
            Path to save the latest epoch's checkpoint to. If this or `best_path`
            is unspecified, no checkpoint can be saved (dry-run).
        best_path: str, optional
            Path to separately save the best-performing epoch's checkpoint to.
            If this or `path` is unspecified, no checkpoint can be saved
            (dry-run).
        resume_from : str, optional
            (to be implemented)
        dvc : bool
            Whether to add saved checkpoints to Data Version Control.

        Returns
        -------
        val_metrics : numpy.ndarray of size (epoch_count, 4)
            Accumulated validation set metrics over all epochs. Four metrics are
            stored: leaf-level accuracy, leaf-level precision, averaged accuracy
            and averaged precision (over all levels). Models that cannot return
            non-leaf classifications must leave averaged accuracy and precision
            as None.
        """
        pass

    @abstractmethod
    def save(self, path, optim=None, dvc=True):
        """
        Save current instance state to disk.

        It also takes in optimiser state for resuming capability.

        All implementations must take care of running ``dvc add`` themselves.

        Parameters
        ----------
        path : str
             Path to save the checkpoint file to.
        optim : Optional(None, torch.optim.Optimizer)
            The current optimiser instance. Checkpoints also save optimiser
            state for resuming training in the future. If the model does
            not use an optimiser, leave this as None.
        dvc : bool
            Whether to add this checkpoint to Data Version Control.
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

        Parameters
        ----------
        path : str
             Path to the checkpoint file.

        Returns
        -------
        optim_dict : dict, optional
        The state dictionary of the optimiser at that time, which can be loaded
        using `optimizer.load_state_dict()`. If the model does not use an
        optimiser, return None.
        """
        pass

    @abstractmethod
    def export_onnx(self, classifier_path, encoder_path=None):
        """
        Export current instance to ONNX. More than one graph can be created.
        It is recommended that you export the classifier head and the encoder
        separately into the two given paths.

        Parameters
        ----------
        classifier_path: str
            Where to write the classifier head to.
        encoder_path: None
            Where to write the encoder (DistilBERT) to.
        """
        pass

    @abstractmethod
    def export_bento_resources(self, svc_config={}):
        """Export the necessary resources to build a BentoML service of this \
        model.

        Parameters
        ----------
        svc_config: dict
            Additional configuration to pack into the BentoService.

        Returns
        -------
        config: dict
            Evidently configuration data specific to this instance.
        svc: BentoService subclass
            A fully packed BentoService.
        """
        pass
