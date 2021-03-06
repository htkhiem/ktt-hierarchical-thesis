"""Implementation of the DB-BHCN+AWX model."""
import os
from importlib import import_module
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from models import model_pytorch
from utils.hierarchy import PerLevelHierarchy
from utils.encoders.distilbert import get_pretrained, get_tokenizer, \
    export_trained, DistilBertPreprocessor

REFERENCE_SET_FEATURE_POOL = 32
POOLED_FEATURE_SIZE = 24


class AWX(torch.nn.Module):
    """Implementation of the Adjacency Wrapping Matrix layer."""

    def __init__(
        self,
        config,
        hierarchy,  # Relies on existing parent_of information
    ):
        """Construct module."""
        super(AWX, self).__init__()
        # If n <= 0, then use the max-mode. Values less than 0 are meaningless.
        self.n = config['awx_norm']
        self.R = hierarchy.R.transpose(0, 1)
        self.nonlinear = torch.nn.Sigmoid()

    def n_norm(self, x, epsilon=1e-6):
        """Compute norm level with epsilon."""
        return torch.pow(
            torch.clamp(
                torch.sum(
                    torch.pow(x, self.n),
                    -1  # dim
                ),
                epsilon,
                1-epsilon
            ),
            1./self.n
        )

    def forward(self, inputs):
        """Forward leaf level of main classifier network through AWX."""
        output = self.nonlinear(inputs)
        # Stack/duplicate outputs so we have one copy for every class.
        # Each of these copies will go through n_norm, min or max
        # depending on l.
        output = output.unsqueeze(1)
        output = output.expand(-1, self.R.shape[0], -1)
        # Stack/duplicate R matrix to account for minibatch
        # Stacking on first axis so we don't need a separate unsqueeze call
        # input.shape[0] is minibatch size.
        R_batch = self.R.expand(inputs.shape[0], -1, -1)

        if self.n > 1:
            output = self.n_norm(torch.mul(output, R_batch))
        elif self.n > 0:
            # That is, n = 1. In this case the modulus is simply sum.
            output = torch.clamp(
                torch.sum(torch.mul(output, R_batch), 2),
                max=1-1e-4
            )
        else:
            # Only take values, discard indices
            output = torch.max(torch.mul(output, R_batch), 2)[0]
        return output

    def to(self, device=None):
        """
        Move this module to specified device.

        This overloads the default PT module's to() method to additionally
        move the R-matrix along.
        """
        super().to(device)
        if device is not None:
            self.R = self.R.to(device)
        return self


class BHCN_AWX(torch.nn.Module):
    """Implementation of DB-BHCN's classifier with AWX integration."""

    def __init__(
        self,
        input_dim,
        hierarchy,
        config
    ):
        """Construct module."""
        super(BHCN_AWX, self).__init__()

        # Back up some parameters for use in forward()
        self.depth = len(hierarchy.levels)
        self.output_dim = len(hierarchy.classes)
        self.level_sizes = hierarchy.levels
        self.level_offsets = hierarchy.level_offsets
        self.parent_of = hierarchy.parent_of
        self.device = config['device']

        # Back up for save/export
        self.hierarchy = hierarchy

        # First layer only takes in BERT encodings
        self.fc_layers = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, hierarchy.levels[0])
        ])
        torch.nn.init.xavier_uniform_(self.fc_layers[0].weight)
        self.norms = torch.nn.ModuleList([])
        for i in range(1, self.depth):
            self.fc_layers.extend([
                torch.nn.Linear(
                    input_dim + hierarchy.levels[i-1],
                    hierarchy.levels[i]
                )
            ])
            torch.nn.init.xavier_uniform_(self.fc_layers[i].weight)
            self.norms.extend([
                torch.nn.LayerNorm(hierarchy.levels[i-1],
                                   elementwise_affine=False)
            ])
        # Activation functions
        self.hidden_nonlinear = (
            torch.nn.ReLU()
            if config['hidden_nonlinear'] == 'relu'
            else torch.nn.Tanh()
        )
        self.output_nonlinear = torch.nn.LogSoftmax(dim=1)

        # AWX layer
        self.awx = AWX(config, hierarchy)

        # Dropout
        self.dropout = torch.nn.Dropout(p=config['dropout'])

    def forward(self, x):
        """Forward-propagate input to generate classification."""
        # We have |D| of these
        local_outputs = torch.zeros((x.shape[0], self.output_dim)).to(
            self.device)
        output_l1 = self.fc_layers[0](self.dropout(x))
        local_outputs[
            :,
            0:self.level_offsets[1]
        ] = self.output_nonlinear(output_l1)

        prev_output = output_l1
        for i in range(1, self.depth):
            output_li = self.fc_layers[i](torch.cat([
                self.dropout(
                    self.norms[i-1](self.hidden_nonlinear(prev_output))
                ), x
            ], dim=1))
            local_outputs[
                :,
                self.level_offsets[i]:self.level_offsets[i + 1]
            ] = self.output_nonlinear(output_li)
            prev_output = output_li

        # prev_output now contains the last hidden layer's output.
        # Pass it raw (un-ReLUed) to AWX
        awx_output = self.awx(prev_output)
        return local_outputs, awx_output

    def to(self, device=None):
        """
        Move this module to specified device.

        This overloads the default PT module's to() method to additionally
        move the AWX layer and other components along.
        """
        super().to(device)
        if device is not None:
            self.device = device
            self.awx = self.awx.to(device)
            self.hierarchy = self.hierarchy.to(device)
        return self


class DB_BHCN_AWX(model_pytorch.PyTorchModel):
    """The whole DB-BHCN+AWX model, DistilBERT included."""

    def __init__(
        self,
        hierarchy,
        config,
    ):
        """Construct the DB-BHCN+AWX model.

        Parameters
        ----------
        hierarchy : PerLevelHierarchy
            A `PerLevelHierarchy` instance to build the model on. The instance
            in question must have the `M`-matrix field computed.
        config : dict
            A configuration dictionary. See the corresponding docs section for
            fields used by this model.
        """
        super(DB_BHCN_AWX, self).__init__()
        self.encoder = get_pretrained()
        # Pooling layer for reference set generation
        self.pool = torch.nn.AvgPool1d(REFERENCE_SET_FEATURE_POOL)
        self.classifier = BHCN_AWX(
            768,
            hierarchy,
            config
        )
        self.config = config
        self.device = 'cpu'  # default
        self.pool = torch.nn.AvgPool1d(REFERENCE_SET_FEATURE_POOL)

    @classmethod
    def get_preprocessor(cls, config):
        """Return a DistilBERT preprocessor instance for this model."""
        return DistilBertPreprocessor(config)

    @classmethod
    def from_checkpoint(cls, path):
        """
        Construct model from saved checkpoint.

        Parameters
        ----------
        path : str
            Path to the checkpoint. Checkpoints have a `.pt` extension.

        Returns
        -------
        instance : DB_BHCN_AWX
            An instance that fully replicates the one producing the checkpoint.

        See also
        --------
        save : Create a checkpoint readable by this method.
        load : An alternative to this method for already-constructed instances.
        """
        checkpoint = torch.load(path)
        hierarchy = PerLevelHierarchy.from_dict(checkpoint['hierarchy'])
        instance = cls(
            hierarchy,
            checkpoint['config'],
        )
        instance.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        instance.classifier.load_state_dict(
            checkpoint['classifier_state_dict']
        )
        return instance

    def forward(self, ids, mask):
        """Forward-propagate input to generate classification, with AWX.

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
        local_outputs : List of torch.FloatTensors
            Classification scores within (0, 1). There are n tensors in the
            list where n is the number of layers in the hierarchy. Each
            tensor is of size (batch_size, level_class_count). To extract the
            predicted classes, one can `argmax` each tensor in their last
            dimension.
        awx_outputs : torch.FloatTensor of size (batch_size, class_count)
            The same output passed through the AWX layer. This is the output
            to use as the final answer of the model.
        """
        local_outputs, awx_outputs = self.classifier(
            self.encoder(ids, mask)[0][:, 0, :]
        )
        # Split local outputs
        local_outputs = [
            local_outputs[
                :,
                self.classifier.level_offsets[i]:
                self.classifier.level_offsets[i+1]]
            for i in range(self.classifier.depth)
        ]
        return local_outputs, awx_outputs

    def forward_with_features(self, ids, mask):
        """Forward-propagate input to generate classification, with AWX.

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
        awx_outputs : torch.FloatTensor of size (batch_size, leaf_class_count)
            AWX'ed scores for the leaf layer. Only the leaf layer is returned
            to as Evidently doesn't seem to play well with multilabel tasks.
        pooled_features: torch.FloatTensor of shape(batch_size, 24)
        """
        encoder_outputs = self.encoder(ids, mask)[0][:, 0, :]
        _, awx_outputs = self.classifier(
            encoder_outputs
        )
        return awx_outputs[
            :,
            self.classifier.level_offsets[-2]:
        ], self.pool(encoder_outputs)

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
            'optimizer': optim
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
        """Train the AWX-equipped variant of DB-BHCN.

        Either train_bhcn or train_bhcn_awx will be called, depending on
        whether this model instance was configured with AWX or not.

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
        resume_from : str, optional
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
        optimizer = torch.optim.Adam(
            [
                {
                    'params': self.encoder.parameters(),
                    'lr': self.config['encoder_lr']
                },
                {
                    'params': self.classifier.parameters(),
                    'lr': self.config['classifier_lr']
                }
            ],
        )

        gamma_L = self.config['gamma_l']
        deviations = np.linspace(
            -gamma_L,
            gamma_L,
            self.classifier.depth
        )
        loss_L_weights = [1] * self.classifier.depth
        loss_L_weights -= deviations
        val_loss_min = np.Inf
        lambda_L = self.config['lambda_l']

        criterion_g = torch.nn.BCELoss()
        criterion_l = torch.nn.NLLLoss()

        optimizer = torch.optim.Adam(
            [
                {
                    'params': self.encoder.parameters(),
                    'lr': self.config['encoder_lr']
                },
                {
                    'params': self.classifier.parameters(),
                    'lr': self.config['classifier_lr']
                }
            ],
        )

        # Store validation metrics after each epoch
        val_metrics = np.empty((4, 0), dtype=float)

        tqdm_disabled = 'progress' in self.config.keys() and not self.config['progress']
        for epoch in range(1, self.config['epoch'] + 1):
            train_loss = 0
            self.train()
            print('Epoch {}: Training'.format(epoch))
            for batch_idx, data in enumerate(tqdm(train_loader, disable=tqdm_disabled)):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                targets = data['labels']
                targets_b = model_pytorch.get_hierarchical_one_hot(
                    targets, self.classifier.hierarchy.levels
                ).to(self.device, dtype=torch.float)

                local_outputs, awx_output = self.forward(ids, mask)

                # We have two loss functions:
                # (l)ocal (per-level), and
                # (g)lobal.
                loss_g = criterion_g(awx_output, targets_b)
                loss_l = lambda_L * sum([
                    criterion_l(
                        local_outputs[level].cpu(),
                        targets[:, level]
                    ) * loss_L_weights[level]
                    for level in range(self.classifier.depth)
                ])

                loss = loss_g + loss_l

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                train_loss = train_loss + (loss.item() - train_loss) / (
                    batch_idx + 1)

            print('Epoch {}: Validating'.format(epoch))
            self.eval()
            val_loss = 0

            val_targets = np.empty((0, self.classifier.depth), dtype=int)
            val_outputs = [np.empty(
                (0, self.classifier.level_sizes[level]),
                dtype=float) for level in range(self.classifier.depth)]

            with torch.no_grad():
                for batch_idx, data in enumerate(tqdm(val_loader, disable=tqdm_disabled)):
                    ids = data['ids'].to(self.device,
                                         dtype=torch.long)
                    mask = data['mask'].to(self.device,
                                           dtype=torch.long)
                    targets = data['labels']
                    targets_b = model_pytorch.get_hierarchical_one_hot(
                        targets, self.classifier.hierarchy.levels
                    ).to(self.device, dtype=torch.float)

                    local_outputs, awx_output = self.forward(ids, mask)

                    # We have two loss functions:
                    # (l)ocal (per-level), and
                    # (g)lobal.
                    loss_g = criterion_g(awx_output, targets_b)
                    loss_l = lambda_L * sum([
                        criterion_l(
                            local_outputs[level].cpu(),
                            targets[:, level]
                        ) * loss_L_weights[level]
                        for level in range(self.classifier.depth)
                    ])
                    loss = loss_g + loss_l

                    val_loss = val_loss + (loss.item() - val_loss) / (
                        batch_idx + 1)

                    val_targets = np.concatenate([
                        val_targets, targets.cpu().detach().numpy()
                    ])

                    # Split AWX output into levels
                    awx_outputs = [
                        awx_output[
                            :,
                            self.classifier.level_offsets[i]:
                            self.classifier.level_offsets[i + 1]
                        ] for i in range(self.classifier.depth)
                    ]

                    for i in range(len(val_outputs)):
                        val_outputs[i] = np.concatenate([
                            val_outputs[i],
                            awx_outputs[i].cpu().detach().numpy()
                        ])

            val_metrics = np.concatenate([
                val_metrics,
                np.expand_dims(
                    model_pytorch.get_metrics(
                        {'outputs': val_outputs, 'targets': val_targets},
                        display='print'
                    ),
                    axis=1
                )],
                axis=1
            )

            train_loss = train_loss/len(train_loader)
            val_loss = val_loss/len(val_loader)

            if path is not None and best_path is not None:
                optim = optimizer.state_dict()
                self.save(path, optim, dvc)
                if val_loss <= val_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}). '
                          'Saving best model...'.format(
                              val_loss_min, val_loss))
                    val_loss_min = val_loss
                    self.save(best_path, optim)
            print('Epoch {}: Done\n'.format(epoch))
        return val_metrics

    def test(self, loader, return_features=False):
        """Test the AWX variant on a dataset.

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

        all_targets = np.empty((0, self.classifier.depth), dtype=int)
        all_outputs = [np.empty(
            (0, self.classifier.level_sizes[level]),
            dtype=float
        ) for level in range(self.classifier.depth)]

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(loader)):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                targets = data['labels']

                _, awx_output = self.forward(ids, mask)

                # Cut AWX outputs to levels
                awx_outputs = [
                    awx_output[
                        :,
                        self.classifier.level_offsets[i]:
                        self.classifier.level_offsets[i + 1]
                    ] for i in range(self.classifier.depth)
                ]

                all_targets = np.concatenate([all_targets, targets])
                for i in range(len(all_outputs)):
                    all_outputs[i] = np.concatenate([
                        all_outputs[i],
                        awx_outputs[i].cpu().detach().numpy()
                    ])

        return {
            'targets': all_targets,
            'outputs': all_outputs,
        }

    def gen_reference_set(self, loader):
        """Generate an Evidently-compatible reference dataset, AWX-enabled.

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

                awx_outputs, pooled_features = self.\
                    forward_with_features(ids, mask)
                all_pooled_features = np.concatenate(
                    [all_pooled_features, pooled_features.cpu()]
                )
                # Only store leaves
                all_targets = np.concatenate([all_targets, targets[:, -1]])
                all_outputs = np.concatenate([all_outputs, awx_outputs.cpu()])

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
        svc_lts = import_module('models.db_bhcn_awx.bentoml.svc_lts')
        svc = svc_lts.DB_BHCN_AWX()
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
        """
        Move this module to specified device.

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
