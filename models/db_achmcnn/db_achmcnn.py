"""Implementation of the Adapted C-HMCNN classifier atop DistilBERT."""
import os
from importlib import import_module

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from models import model_pytorch
from utils.hierarchy import PerLevelHierarchy
from utils.encoders.distilbert import get_pretrained, get_tokenizer,\
    export_trained, DistilBertPreprocessor

REFERENCE_SET_FEATURE_POOL = 32
POOLED_FEATURE_SIZE = 768 // REFERENCE_SET_FEATURE_POOL


class MCM(torch.nn.Module):
    """Implementation of the MCM post-processing layer."""

    def __init__(self, M):
        """Construct module."""
        super(MCM, self).__init__()
        self.M = M

    def forward(self, x):
        """Post-process normal FC outputs to be hierarchically-compliant."""
        n = self.M.shape[1]
        H = x.unsqueeze(1)  # Add a new dimension
        # Duplicate x along the new dimension to create a list of 2D matrices
        # of size n x n (same as R). Note that x can be a list of vectors
        # instead of one.
        H = H.expand(x.shape[0], n, n)
        # We'll have to duplicate R to multiply with the entire batch here
        M_batch = self.M.expand(x.shape[0], n, n)
        final_out, _ = torch.max(M_batch*H, dim=2)
        return final_out

    def to(self, device=None):
        """
        Move this module to specified device.

        This overloads the default PT module's to() method to additionally
        move the M-matrix along.
        """
        super().to(device)
        if device is not None:
            self.M = self.M.to(device)
        return self


class H_MCM_Model(torch.nn.Module):
    """Implementation of a standard FC network coupled with the MCM layer."""

    def __init__(self, input_dim, hierarchy, config):
        """Construct module."""
        super(H_MCM_Model, self).__init__()

        self.depth = len(hierarchy.levels)
        self.level_sizes = hierarchy.levels
        self.level_offsets = hierarchy.level_offsets
        self.layer_count = config['h_layer_count']
        self.mcm = MCM(hierarchy.M)

        # Back up the hierarchy object for exporting
        self.hierarchy = hierarchy

        output_dim = len(hierarchy.classes)

        fc = []
        if self.layer_count == 1:
            fc.append(torch.nn.Linear(input_dim, output_dim))
        else:
            for i in range(self.layer_count):
                if i == 0:
                    fc.append(torch.nn.Linear(
                        input_dim,
                        config['h_hidden_dim']
                    ))
                elif i == self.layer_count - 1:
                    fc.append(torch.nn.Linear(
                        config['h_hidden_dim'],
                        output_dim
                    ))
                else:
                    fc.append(torch.nn.Linear(
                        config['h_hidden_dim'],
                        config['h_hidden_dim']
                    ))

        self.fc = torch.nn.ModuleList(fc)
        self.drop = torch.nn.Dropout(config['h_dropout'])
        self.sigmoid = torch.nn.Sigmoid()
        if config['h_nonlinear'] == 'tanh':
            self.f = torch.nn.Tanh()
        else:
            self.f = torch.nn.ReLU()

    def forward(self, x):
        """Forward-propagate input to generate classification."""
        for i in range(self.layer_count):
            if i == self.layer_count - 1:
                x = self.sigmoid(self.fc[i](x))
            else:
                x = self.f(self.fc[i](x))
                x = self.drop(x)
        if self.training:
            return x
        return self.mcm(x)

    def to(self, device=None):
        """
        Move this module to specified device.

        This overloads the default PT module's to() method to additionally
        move the MCM layer along.
        """
        super().to(device)
        if device is not None:
            self.mcm = self.mcm.to(device)
            self.hierarchy = self.hierarchy.to(device)
        return self


class DB_AC_HMCNN(model_pytorch.PyTorchModel, torch.nn.Module):
    """Wrapper class combining DistilBERT and the adapted C-HMCNN model."""

    def __init__(self, hierarchy, config):
        """Construct the DistilBERT + Adapted C-HMCNN model.

        Parameters
        ----------
        hierarchy : PerLevelHierarchy
            A `PerLevelHierarchy` instance to build the model on. The instance
            in question must have the `M`-matrix field computed.
        config : dict
            A configuration dictionary. See the corresponding docs section for
            fields used by this model.
        """
        super(DB_AC_HMCNN, self).__init__()
        self.encoder = get_pretrained()
        self.classifier = H_MCM_Model(
            768,  # DistilBERT outputs 768 values.
            hierarchy,
            config
        )
        self.config = config
        self.device = 'cpu'

        # For reference set generation
        self.pool = torch.nn.AvgPool1d(REFERENCE_SET_FEATURE_POOL)

    @classmethod
    def get_preprocessor(cls, config):
        """Return a DistilBERT preprocessor instance for this model."""
        return DistilBertPreprocessor(config)

    @classmethod
    def from_checkpoint(cls, path):
        """Construct model from saved checkpoints as produced by previous\
        instances of this model.

        Parameters
        ----------
        path : str
            Path to the checkpoint. Checkpoints have a `.pt` extension.

        Returns
        -------
        instance : DB_AC_HMCNN
            An instance that fully replicates the one producing the checkpoint.

        See also
        --------
        save : Create a checkpoint readable by this method.
        load : An alternative to this method for already-constructed instances.
        """
        checkpoint = torch.load(path)
        hierarchy = PerLevelHierarchy.from_dict(checkpoint['hierarchy'])
        instance = cls(hierarchy, checkpoint['config'])
        instance.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        instance.classifier.load_state_dict(
            checkpoint['classifier_state_dict']
        )
        return instance

    def forward(self, ids, mask):
        """Forward-propagate tokeniser input to generate classification.

        This model takes in the `ids` and `mask` tensors as generated by a
        `DistilBertTokenizer` or `DistilBertTokenizerFast` instance.

        Parameters
        ----------
        ids : torch.LongTensor of shape (batch_size, num_choices)
             Indices of input sequence tokens in the vocabulary.
        mask : torch.FloatTensor of shape (batch_size, num_choices), optional
            Mask to avoid performing attention on padding token indices. Mask
            values are 0 for real tokens and 1 for masked tokens such as pads.

        Returns
        -------
        scores : torch.FloatTensor of shape (batch_size, class_count)
            Classification scores within (0, 1). Classes are ordered by their
            hierarchical level. To extract the predicted classes, one can
            `argmax` ranges of the second dimension corresponding to each
            level.
        """
        return self.classifier(self.encoder(ids, mask)[0][:, 0, :])

    def save(self, path, optim, dvc=True):
        """Save model state to disk using PyTorch's pickle facilities.

        The model state is saved as a `.pt` checkpoint with all the weights
        as well as supplementary data required to reconstruct this exact
        instance, including its topology. See the related docs section for
        the schema of checkpoints produced by this model.

        This method is mostly used internally but could be of use in case
        one prefers to implement a custom training routine.

        Parameters
        ----------
        path : str
             Path to save the checkpoint file to.
        optim : torch.optim.Optimizer
            The current optimiser instance. Checkpoints also save optimiser
            state for resuming training in the future.
        dvc : bool
            Whether to add this checkpoint to Data Version Control.
        """
        checkpoint = {
            'config': self.config,
            'hierarchy': self.classifier.hierarchy.to_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': optim
        }
        torch.save(checkpoint, path)

        if dvc:
            os.system('dvc add ' + path)

    def load(self, path):
        """Load model state from disk.

        Unlike `from_checkpoint`, this method does not alter the instance's
        topology and as such can only be used with checkpoints whose topology
        matches exactly with this instance. In other words, if you use this
        method with a checkpoint, you have to ensure that the current instance's
        topology matches that of the past instance that saved said checkpoint.
        This method is useful for loading a previous state of the same instance
        for benchmarking or continuing training, as it not only loads weights
        but also returns the previous optimiser state.

        Parameters
        ----------
        path : str
             Path to the checkpoint file.

        Returns
        -------
        optim_dict : dict
        The state dictionary of the optimiser at that time, which can be loaded
        using `optimizer.load_state_dict()`.
        """
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        return checkpoint['optimizer_state_dict']

    def fit(
            self,
            train_loader,
            val_loader,
            path=None,
            best_path=None,
            resume_from=None,
            dvc=True
    ):
        """Train this DistilBERT + Adapted C-HMCNN instance.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            A minibatched, shuffled PyTorch DataLoader containing the training
            set.
        val_loader : torch.utils.data.DataLoader
            A minibatched, shuffled PyTorch DataLoader containing the validation
            set.
        path : str, optional
            Path to save the latest epoch's checkpoint to. If this or `best_path`
            is unspecified, no checkpoint will be saved (dry-run).
        best_path: str, optional
            Path to separately save the best-performing epoch's checkpoint to.
            If this or `path` is unspecified, no checkpoint will be saved
            (dry-run).
        resume_from: str, optional
            (to be implemented)
        dvc : bool
            Whether to add saved checkpoints to Data Version Control.

        Returns
        -------
        val_metrics : numpy.ndarray of size (epoch_count, 4)
            Accumulated validation set metrics over all epochs. Four metrics are
            stored: leaf-level accuracy, leaf-level precision, averaged accuracy
            and averaged precision (over all levels).
        """
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            [
                {
                    'params': self.encoder.parameters(),
                    'lr': self.config['encoder_lr'],
                },
                {
                    'params': self.classifier.parameters(),
                    'lr': self.config['classifier_lr']
                }
            ],
        )
        tqdm_disabled = 'progress' in self.config.keys() and not self.config['progress']
        val_loss_min = np.Inf
        # Store validation metrics after each epoch
        val_metrics = np.empty((4, 0), dtype=float)
        for epoch in range(1, self.config['epoch'] + 1):
            train_loss = 0
            val_loss = 0
            self.train()
            print('Epoch {}: Training'.format(epoch))
            for batch_idx, data in enumerate(tqdm(train_loader, disable=tqdm_disabled)):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                targets = data['labels']
                # Convert targets to one-hot
                targets_b = model_pytorch.get_hierarchical_one_hot(
                    targets, self.classifier.hierarchy.levels
                ).to(self.device, dtype=torch.float)
                targets = targets.to(self.device, dtype=torch.float)

                outputs = self.forward(ids, mask)

                # Notation: H = output stacked, Hbar = hbar stacked
                # MCM = max(M * H, dim=1)
                constr_outputs = self.classifier.mcm(outputs)
                # hbar = y * h
                train_outputs = targets_b * outputs.float()
                # max(M * Hbar, dim = 1)
                train_outputs = self.classifier.mcm(train_outputs)

                # (1-y) + max(M * H, dim = 1) + y * max(M * Hbar, dim = 1)
                train_outputs = (1-targets_b)*constr_outputs.float() + (
                    targets_b*train_outputs
                )
                loss = criterion(train_outputs, targets_b)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                train_loss = train_loss + ((1 / (batch_idx + 1)) * (
                    loss.item() - train_loss
                ))

            print('Epoch {}: Validating'.format(epoch))
            self.eval()

            val_targets = np.empty((0, self.classifier.depth), dtype=int)
            val_outputs = [
                np.empty(
                    (0, self.classifier.level_sizes[level]),
                    dtype=float)
                for level in range(self.classifier.depth)
            ]

            # Validation
            with torch.no_grad():
                for batch_idx, data in enumerate(tqdm(val_loader, disable=tqdm_disabled)):
                    ids = data['ids'].to(self.device,
                                         dtype=torch.long)
                    mask = data['mask'].to(self.device,
                                           dtype=torch.long)
                    targets = data['labels']
                    # Convert targets to one-hot
                    targets_b = model_pytorch.get_hierarchical_one_hot(
                        targets, self.classifier.hierarchy.levels
                    ).to(self.device, dtype=torch.float)

                    constrained_outputs = self.forward(ids, mask).float()

                    loss = criterion(constrained_outputs, targets_b)

                    # Split local outputs
                    local_outputs = [
                        constrained_outputs[
                            :,
                            self.classifier.level_offsets[i]:
                            self.classifier.level_offsets[i+1]
                        ] for i in range(self.classifier.depth)
                    ]

                    val_loss = val_loss + ((1 / (batch_idx + 1)) * (
                        loss.item() - val_loss
                    ))

                    val_targets = np.concatenate([
                        val_targets,
                        targets])
                    for i in range(len(val_outputs)):
                        val_outputs[i] = np.concatenate([
                            val_outputs[i],
                            local_outputs[i].cpu().detach().numpy()
                        ])

                train_loss = train_loss/len(train_loader)
                val_loss = val_loss/len(val_loader)

                val_metrics = np.concatenate(
                    [
                        val_metrics,
                        np.expand_dims(
                            model_pytorch.get_metrics({
                                'outputs': val_outputs,
                                'targets': val_targets
                            }, display='print'), axis=1
                        )
                    ],
                    axis=1
                )

                if path is not None and best_path is not None:
                    optim = optimizer.state_dict()
                    self.save(path, optim, dvc)
                    if val_loss <= val_loss_min:
                        print('Validation loss decreased ({:.6f} --> {:.6f}). '
                              'Saving best model...'.format(
                                  val_loss_min, val_loss
                              ))
                        val_loss_min = val_loss
                        self.save(best_path, optim)
            print('Epoch {}: Done\n'.format(epoch))
        return val_metrics

    def test(self, loader):
        """Test this model on a dataset.

        This method can be used to run this instance (trained or not) over any
        dataset wrapped in a suitable PyTorch DataLoader. No gradient descent
        or backpropagation will take place.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            A minibatched, shuffled PyTorch DataLoader containing the training
            set.

        Returns
        -------
        test_metrics : numpy.ndarray of size (epoch_count, 4)
            Accumulated validation set metrics over all epochs. Four metrics
            are stored: leaf-level accuracy, leaf-level precision, averaged
            accuracy and averaged precision (over all levels).
        """
        self.eval()

        all_targets = np.empty((0, self.classifier.depth), dtype=bool)
        all_outputs = [
            np.empty(
                (0, self.classifier.level_sizes[level]),
                dtype=float
            )
            for level in range(self.classifier.depth)
        ]

        # Validation
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(loader)):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                targets = data['labels']

                constrained_outputs = self.forward(ids, mask).float()
                # Split local outputs
                local_outputs = [
                    constrained_outputs[
                        :,
                        self.classifier.level_offsets[i]:
                        self.classifier.level_offsets[i+1]
                    ]
                    for i in range(self.classifier.depth)
                ]

                all_targets = np.concatenate([all_targets, targets])
                for i in range(len(all_outputs)):
                    all_outputs[i] = np.concatenate([
                        all_outputs[i],
                        local_outputs[i].cpu().detach().numpy()
                    ])

        return {
            'targets': all_targets,
            'outputs': all_outputs,
        }

    def forward_with_features(self, ids, mask):
        """Forward-propagate input to generate classification.

        This version additionally returns pooled features from DistilBERT
        for generating the reference set. By default, a kernel size and
        stride of 32 is used, which means there are 24 features pooled from
        the original 768 features.

        Parameters
        ----------
        ids : torch.LongTensor of shape (batch_size, num_choices)
             Indices of input sequence tokens in the vocabulary.
        mask : torch.FloatTensor of shape (batch_size, num_choices), optional
            Mask to avoid performing attention on padding token indices. Mask
            values are 0 for real tokens and 1 for masked tokens such as pads.

        Returns
        -------
        leaf_outputs : torch.FloatTensor of size (batch_size, leaf_class_count)
            Scores for the leaf layer. Only the leaf layer is returned
            to as Evidently doesn't seem to play well with multilabel tasks.
        pooled_features: torch.FloatTensor of shape(batch_size, 24)
        """
        encoder_outputs = self.encoder(ids, mask)[0][:, 0, :]
        outputs = self.classifier(
            encoder_outputs
        )
        return outputs[
            :, self.classifier.level_offsets[-2]:], self.pool(encoder_outputs)

    def gen_reference_set(self, loader):
        """Generate an Evidently-compatible reference dataset.

        Due to the data drift tests' computational demands, we only record
        average-pooled features.

        Parameters
        ----------
        loader: torch.utils.data.DataLoader
            A DataLoader wrapping around a dataset to become the reference
            dataset (ideally the test set).

        Returns
        -------
        reference_set: pandas.DataFrame
            An Evidently-compatible dataset. Numerical feature column names are
            simple stringified numbers from 0 to 23 (for the default kernel/
            stride of 32), while targets are leaf classes' string names.
        """
        self.eval()
        all_pooled_features = np.empty((0, POOLED_FEATURE_SIZE))
        all_targets = np.empty((0), dtype=int)
        all_outputs = np.empty(
            (0, self.classifier.hierarchy.levels[-1]), dtype=float)

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(loader)):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                targets = data['labels']

                leaf_outputs, pooled_features = self.\
                    forward_with_features(ids, mask)
                all_pooled_features = np.concatenate(
                    [all_pooled_features, pooled_features.cpu()]
                )
                # Only store leaves
                all_targets = np.concatenate([all_targets, targets[:, -1]])
                all_outputs = np.concatenate([all_outputs, leaf_outputs.cpu()])

        cols = {
            'targets': all_targets
        }
        leaf_start = self.classifier.hierarchy.level_offsets[-2]
        for col_idx in range(all_pooled_features.shape[1]):
            cols[str(col_idx)] = all_pooled_features[:, col_idx]
        for col_idx in range(all_outputs.shape[1]):
            cols[
                self.classifier.hierarchy.classes[leaf_start + col_idx]
            ] = all_outputs[:, col_idx]
        return pd.DataFrame(cols)

    def export_onnx(self, classifier_path, encoder_path=None):
        """Export this model as two ONNX graphs.

        Parameters
        ----------
        classifier_path: str
            Where to write the classifier head to.
        encoder_path: None
            Where to write the encoder (DistilBERT) to.
        """
        self.eval()
        if encoder_path is None:
            raise RuntimeError('This model requires an encoder path')
        export_trained(
            self.encoder,
            encoder_path
        )
        x = torch.randn(1, 768, requires_grad=True).to(
            self.device
        )
        # Export into transformers model .bin format
        torch.onnx.export(
            self.classifier,
            x,
            classifier_path + 'classifier.onnx',
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        self.classifier.hierarchy.to_json(
            "{}/hierarchy.json".format(classifier_path)
        )

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
        self.eval()
        # Sample input
        x = torch.randn(1, 768, requires_grad=True).to(self.device)
        # Config for monitoring service
        config = {
            'prediction': self.classifier.hierarchy.classes[
                self.classifier.hierarchy.level_offsets[-2]:
                self.classifier.hierarchy.level_offsets[-1]
            ]
        }
        svc_lts = import_module('models.db_achmcnn.bentoml.svc_lts')
        svc = svc_lts.DB_AC_HMCNN()
        # Pack tokeniser along with encoder
        encoder = {
            'tokenizer': get_tokenizer(),
            'model': self.encoder
        }
        svc.pack('encoder', encoder)
        svc.pack('classifier', torch.jit.trace(self.classifier, x))
        svc.pack('hierarchy', self.classifier.hierarchy.to_dict())
        svc.pack('config', svc_config)
        return config, svc

    def to(self, device=None):
        """Move this module to specified device.

        This overloads the default PT module's to() method to additionally
        set its internal device variable and moves its submodules.

        Parameters
        ----------
        device : torch.device ,optional
            The device to move the model to.

        Returns
        -------
        self
        """
        super().to(device)
        if device is not None:
            self.encoder.to(device)
            self.classifier.to(device)
            self.device = device
        return self


if __name__ == "__main__":
    pass
