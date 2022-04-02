"""Implementation of the Adapted C-HMCNN classifier atop DistilBERT."""
import os

import torch
import numpy as np
from tqdm import tqdm
import bentoml

from models import model, model_pytorch
from utils.hierarchy import PerLevelHierarchy
from utils.distilbert import get_pretrained, export_trained


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


class DB_AC_HMCNN(model.Model, torch.nn.Module):
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

    @classmethod
    def from_checkpoint(cls, path):
        """Construct model from saved checkpoints as produced by previous
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
        if not os.path.exists(path):
            if not os.path.exists(path + '.dvc'):
                raise OSError('Checkpoint not present and cannot be retrieved')
            os.system('dvc checkout {}.dvc'.format(path))
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
        val_loss_min = np.Inf
        # Store validation metrics after each epoch
        val_metrics = np.empty((4, 0), dtype=float)
        for epoch in range(1, self.config['epoch'] + 1):
            train_loss = 0
            val_loss = 0
            self.train()
            print('Epoch {}: Training'.format(epoch))
            for batch_idx, data in enumerate(tqdm(train_loader)):
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
                for batch_idx, data in enumerate(tqdm(val_loader)):
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

    def export(self, dataset_name, bento=False):
        """Export model to ONNX/Bento.

        The classifier head (AC-HMCNN) is always exported as ONNX (which is
        then loaded into BentoML as an `onnxruntime` runner). The DistilBERT
        encoder is either exported as ONNX or straight to BentoML.

        Parameters
        ----------
        dataset_name: str
            Name of the dataset this instance was trained on. Use the folder
            name of the intermediate version in the datasets folder.

        bento: bool
            Whether to export this model as a BentoML model or not. If true,
            the DistilBERT encoder will be directly packaged into a BentoML
            model and the entire model will be saved in your local BentoML
            store. The classifier is always exported as ONNX.
        """
        self.eval()
        export_trained(self.encoder, dataset_name, 'db_achmcnn', bento=bento)

        # Create dummy input for tracing
        batch_size = 1  # Dummy batch size. When exported, it will be dynamic
        x = torch.randn(batch_size, 768, requires_grad=True).to(
            self.device
        )
        name = '{}_{}'.format('db_achmcnn', dataset_name)
        path = 'output/{}/classifier/'.format(name)

        if not os.path.exists(path):
            os.makedirs(path)

        path += 'classifier.onnx'

        # Clear previous versions
        if os.path.exists(path):
            os.remove(path)

        # Export into transformers model .bin format
        torch.onnx.export(
            self.classifier,
            x,
            path,
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

        hierarchy_json = self.classifier.hierarchy.to_json(
            "output/{}/hierarchy.json".format(name)
        )

        # Optionally save to BentoML model store. Pack hierarchical metadata
        # along with model for convenience.
        if bento:
            bentoml.onnx.save(
                'classifier_' + name,
                path,
                metadata=hierarchy_json
            )

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
