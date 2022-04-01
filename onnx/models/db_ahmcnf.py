"""Implementation of the Adapted HMCN-F classifier atop DistilBERT."""
import os

import torch
import numpy as np
from tqdm import tqdm
import bentoml

from models import model
from utils.hierarchy import PerLevelHierarchy
from utils.distilbert import get_pretrained
from utils.dataset import get_hierarchical_one_hot
from utils.metric import get_metrics


class AHMCN_F(torch.nn.Module):
    """Adapted implementation of the HMCN-F classifier model."""

    def __init__(
        self,
        input_dim,
        hierarchy,
        config
    ):
        """Construct module."""
        super(AHMCN_F, self).__init__()

        # Back up some parameters for use in forward()
        self.depth = len(hierarchy.levels)
        self.global_weight = config['global_weight']
        self.output_dim = len(hierarchy.classes)
        self.level_sizes = hierarchy.levels
        self.level_offsets = hierarchy.level_offsets
        self.parent_of = hierarchy.parent_of
        self.device = config['device']

        # Back up the hierarchy object for exporting
        self.hierarchy = hierarchy

        # Construct global layers (main flow)
        global_layers = []
        global_layer_norms = []
        for i in range(self.depth):
            if i == 0:
                global_layers.append(torch.nn.Linear(
                    input_dim,
                    config['global_hidden_sizes'][0]
                ))
            else:
                global_layers.append(torch.nn.Linear(
                    config['global_hidden_sizes'][i-1] + input_dim,
                    config['global_hidden_sizes'][i]
                ))
            global_layer_norms.append(torch.nn.LayerNorm(
                config['global_hidden_sizes'][i]
            ))
        self.global_layers = torch.nn.ModuleList(global_layers)
        self.global_layer_norms = torch.nn.ModuleList(global_layer_norms)
        # Global prediction layer
        self.global_prediction_layer = torch.nn.Linear(
            config['global_hidden_sizes'][-1] + input_dim,
            len(hierarchy.classes)
        )

        # Construct local branches (local flow).
        # Each local branch has two linear layers: a transition layer and a
        # local classification layer
        transition_layers = []
        local_layer_norms = []
        local_layers = []

        for i in range(self.depth):
            transition_layers.append(torch.nn.Linear(
                config['global_hidden_sizes'][i],
                config['local_hidden_sizes'][i]
            ))
            local_layer_norms.append(
                torch.nn.LayerNorm(config['local_hidden_sizes'][i])
            )
            local_layers.append(torch.nn.Linear(
                config['local_hidden_sizes'][i],
                hierarchy.levels[i]
            ))
            self.local_layer_norms = torch.nn.ModuleList(local_layer_norms)
            self.transition_layers = torch.nn.ModuleList(transition_layers)
            self.local_layers = torch.nn.ModuleList(local_layers)

        # Activation functions
        self.hidden_nonlinear = (
            torch.nn.ReLU()
            if config['hidden_nonlinear'] == 'relu'
            else torch.nn.Tanh()
        )
        self.output_nonlinear = torch.nn.Sigmoid()

        # Dropout
        self.dropout = torch.nn.Dropout(p=config['dropout'])

    def forward(self, x):
        """Forward-propagate input to generate classification."""
        # We have |D| hidden layers plus one global prediction layer
        local_outputs = torch.zeros((x.shape[0], self.output_dim)).to(self.device)
        output = x  # Would be global path output until the last step
        for i in range(len(self.global_layers)):
            # Global path
            if i == 0:
                # Don't concatenate x into the first layer's input
                output = self.hidden_nonlinear(
                    self.global_layer_norms[i](
                        self.global_layers[i](output)
                    )
                )
            else:
                output = self.hidden_nonlinear(self.global_layer_norms[i](
                    self.global_layers[i](torch.cat([output, x], dim=1))
                ))

            # Local path. Note the dropout between the transition ReLU layer
            # and the local layer.
            local_output = self.dropout(
                self.hidden_nonlinear(
                    self.local_layer_norms[i](self.transition_layers[i](output))
                )
            )
            local_output = self.output_nonlinear(
                self.local_layers[i](local_output))
            local_outputs[
                :,
                self.level_offsets[i]:self.level_offsets[i + 1]] = local_output

            # Dropout main flow for next layer
            output = self.dropout(output)

        global_outputs = self.output_nonlinear(
            self.global_prediction_layer(torch.cat([output, x], dim=1))
        )
        output = self.global_weight * global_outputs + (1 - self.global_weight) * local_outputs
        return output, local_outputs

    def to(self, device=None):
        """
        Move this module to specified device.

        This overloads the default PT module's to() method to move internal
        components along.
        """
        super().to(device)
        if device is not None:
            self.device = device
            self.hierarchy = self.hierarchy.to(device)
        return self


class DB_AHMCN_F(model.Model, torch.nn.Module):
    """Wrapper class combining DistilBERT with the adapted HMCN-F classifier model."""

    def __init__(
        self,
        hierarchy,
        config
    ):
        """Construct the DistilBERT + Adapted  HMCN-F model.

        Parameters
        ----------
        hierarchy : PerLevelHierarchy
            A `PerLevelHierarchy` instance to build the model on. The instance
            in question must have the `M`-matrix field computed.
        config : dict
            A configuration dictionary. See the corresponding docs section for
            fields used by this model.
        """
        super(DB_AHMCN_F, self).__init__()
        self.encoder = get_pretrained()
        self.classifier = AHMCN_F(
            768,
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
        output : torch.FloatTensor of shape (batch_size, class_count)

        local_output: torch.FloatTensor of shape (batch_size, class_count)
        
        """
        output, local_outputs = self.classifier(
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
        return output, local_outputs

    def save(self, path, optim):
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
        """
        checkpoint = {
            'config': self.config,
            'hierarchy': self.classifier.hierarchy.to_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': optim
        }
        torch.save(checkpoint, path)

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
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        return checkpoint['optimizer_state_dict']

    def fit(
            self,
            train_loader,
            val_loader,
            path=None,
            best_path=None,
            resume_from=None
    ):
        """Train this DistilBERT + Adapted HMCN-F instance.

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

        Returns
        -------
        val_metrics : numpy.ndarray of size (epoch_count, 4)
            Accumulated validation set metrics over all epochs. Four metrics are
            stored: leaf-level accuracy, leaf-level precision, averaged accuracy
            and averaged precision (over all levels).
        """
        # HMCN-F's implementation uses a global-space vector of
        # parent-class indices.
        global_parent_of = torch.cat(
            self.classifier.parent_of, axis=0).to(self.device)

        # Keep min validation (test set) loss so we can separately back up our
        # best-yet model
        val_loss_min = np.Inf

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            params=self.classifier.parameters(),
            lr=self.config['classifier_lr']
        )

        # Store validation metrics after each epoch
        val_metrics = np.empty((4, 0), dtype=float)

        # Hierarchical loss gain
        lambda_h = self.config['lambda_h']
        for epoch in range(1, self.config['epoch'] + 1):
            train_loss = 0
            val_loss = 0
            self.train()
            print('Epoch {}: Training'.format(epoch))
            for batch_idx, data in enumerate(tqdm(train_loader)):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                targets = data['labels']
                targets_b = get_hierarchical_one_hot(
                    targets, self.classifier.hierarchy.levels
                ).to(self.device, dtype=torch.float)

                output, local_outputs = self.forward(ids, mask)

                optimizer.zero_grad()

                # We have three loss functions: (g)lobal, (l)ocal, and
                # (h)ierarchical.
                loss_g = criterion(output, targets_b)
                loss_l = sum([
                    criterion(
                        local_outputs[level],
                        targets_b[
                            :,
                            self.classifier.level_offsets[level]:
                            self.classifier.level_offsets[level + 1]]
                    ) for level in range(self.classifier.depth)
                ])
                # output_cpu = output.cpu().detach()
                loss_h = torch.sum(lambda_h * torch.clamp(
                    output -
                    output.index_select(1, global_parent_of),
                    min=0) ** 2)
                loss = loss_g + loss_l + loss_h

                loss.backward()
                optimizer.step()

                train_loss = train_loss + (loss.item() - train_loss) / (batch_idx + 1)

            print('Epoch {}: Validating'.format(epoch))
            self.eval()

            val_targets = np.empty((0, self.classifier.depth), dtype=int)
            val_outputs = [
                np.empty(
                    (0, self.classifier.level_sizes[level]),
                    dtype=float
                ) for level in range(self.classifier.depth)]

            with torch.no_grad():
                for batch_idx, data in tqdm(enumerate(val_loader)):
                    ids = data['ids'].to(self.device,
                                         dtype=torch.long)
                    mask = data['mask'].to(self.device,
                                           dtype=torch.long)
                    targets = data['labels']
                    targets_b = get_hierarchical_one_hot(
                        targets, self.classifier.hierarchy.levels
                    ).to(self.device, dtype=torch.float)

                    output, local_outputs = self.forward(ids, mask)

                    loss_g = criterion(output, targets_b)
                    loss_l = sum([
                        criterion(
                            local_outputs[level],
                            targets_b[
                                :,
                                self.classifier.level_offsets[level]:
                                self.classifier.level_offsets[level + 1]
                            ]
                        ) for level in range(self.classifier.depth)])
                    # output_cpu = output.cpu().detach()
                    loss_h = torch.sum(lambda_h * torch.clamp(
                        output -
                        output.index_select(1, global_parent_of),
                        min=0) ** 2)
                    loss = loss_g + loss_l + loss_h

                    val_loss = val_loss + (loss.item() - val_loss) / (
                        batch_idx + 1)

                    val_targets = np.concatenate(
                        [val_targets, targets.cpu().detach().numpy()]
                    )
                    for i in range(len(val_outputs)):
                        val_outputs[i] = np.concatenate([
                            val_outputs[i],
                            local_outputs[i].cpu().detach().numpy()
                        ])

                val_metrics = np.concatenate(
                    [
                        val_metrics,
                        np.expand_dims(
                            get_metrics(
                                {
                                    'outputs': val_outputs,
                                    'targets': val_targets
                                },
                                display='print'), axis=1
                        )
                    ],
                    axis=1
                )
                train_loss = train_loss/len(train_loader)
                val_loss = val_loss/len(val_loader)

                if path is not None and best_path is not None:
                    optim = optimizer.state_dict()
                    self.save(path, optim)
                    if val_loss <= val_loss_min:
                        print('Validation loss decreased ({:.6f} --> {:.6f}). Saving best model...'.format(val_loss_min,val_loss))
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
            Accumulated validation set metrics over all epochs. Four metrics are
            stored: leaf-level accuracy, leaf-level precision, averaged accuracy
            and averaged precision (over all levels).
        """
        self.eval()

        all_targets = np.empty((0, self.classifier.depth), dtype=bool)
        all_outputs = [
            np.empty((0, self.classifier.level_sizes[level]), dtype=float)
            for level in range(self.classifier.depth)
        ]

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(loader)):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                targets = data['labels']

                _, local_outputs = self.forward(ids, mask)

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
        """Export this model to BentoML.

        Parameters
        ----------
        dataset_name: string
            The name of the dataset used to train/test the model.
        bento: bool
            Bento flag, if on then the model will be exported directly
            into BentoML.
        """
        self.eval()
        # HMCN-F does not fine-tune DistilBERT. No need to export.
        # Create dummy input for tracing
        batch_size = 1  # Dummy batch size. When exported, it will be dynamic
        x = torch.randn(batch_size, 768, requires_grad=True).to(
            self.device
        )
        name = '{}_{}'.format('db_ahmcnf', dataset_name)
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
            self.classifier.to(device)
            self.device = device
        return self


if __name__ == "__main__":
    pass
