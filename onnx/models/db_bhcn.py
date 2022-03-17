"""Implementation of the DistilBERT Branching Hierarchical Classification Network."""
import os

import torch
import numpy as np
from tqdm import tqdm
import bentoml

from models import model
from utils.hierarchy import PerLevelHierarchy
from utils.distilbert import get_pretrained, export_trained
from utils.dataset import get_hierarchical_one_hot
from utils.metric import get_metrics


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


class BHCN(torch.nn.Module):
    """Implementation of the classifier part of the DB-BHCN model."""

    def __init__(
        self,
        input_dim,
        hierarchy,
        config,
    ):
        """Construct a BHCN classifier network."""
        super(BHCN, self).__init__()

        # Back up some parameters for use in forward()
        self.depth = len(hierarchy.levels)
        self.output_dim = len(hierarchy.classes)
        self.level_sizes = hierarchy.levels
        self.level_offsets = hierarchy.level_offsets
        self.parent_of = hierarchy.parent_of
        self.device = 'cpu'  # default

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
                torch.nn.LayerNorm(hierarchy.levels[i-1], elementwise_affine=False)
            ])
        # Activation functions
        self.hidden_nonlinear = (
            torch.nn.ReLU()
            if config['hidden_nonlinear'] == 'relu'
            else torch.nn.Tanh()
        )
        self.output_nonlinear = torch.nn.LogSoftmax(dim=1)

        # Dropout
        self.dropout = torch.nn.Dropout(p=config['dropout'])

    def forward(self, x):
        """Forward-propagate input to generate classification."""
        # We have |D| of these
        local_outputs = torch.zeros((x.shape[0], self.output_dim)).to(self.device)
        output_l1 = self.fc_layers[0](self.dropout(x))
        local_outputs[:, 0:self.level_offsets[1]] = self.output_nonlinear(
            output_l1)

        prev_output = self.hidden_nonlinear(output_l1)
        for i in range(1, self.depth):
            output_li = self.fc_layers[i](torch.cat([
                self.dropout(self.norms[i-1](prev_output)), x], dim=1))
            local_outputs[
                :,
                self.level_offsets[i]:self.level_offsets[i + 1]
            ] = self.output_nonlinear(output_li)
            prev_output = self.hidden_nonlinear(output_li)

        return local_outputs

    def to(self, device=None):
        """
        Move this module to specified device.

        This overloads the default PT module's to() method to additionally
        set its internal device flag and move its internal components.
        """
        super().to(device)
        if device is not None:
            self.device = device
            self.hierarchy = self.hierarchy.to(device)
        return self


class BHCN_AWX(torch.nn.Module):
    """Implementation of DB-BHCN's classifier with AWX integration."""

    def __init__(
        self,
        input_dim,
        hierarchy,
        config,
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
                torch.nn.LayerNorm(hierarchy.levels[i-1], elementwise_affine=False)
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


class DB_BHCN(model.Model, torch.nn.Module):
    """The whole DB-BHCN model, DistilBERT included. AWX is optional."""

    def __init__(
        self,
        hierarchy,
        config,
        awx=False,
    ):
        """Construct module."""
        super(DB_BHCN, self).__init__()
        self.encoder = get_pretrained()
        if awx:
            self.classifier = BHCN_AWX(
                768,
                hierarchy,
                config
            )
        else:
            self.classifier = BHCN(
                768,
                hierarchy,
                config
            )
        self.config = config
        self.awx = awx
        self.device = 'cpu'  # default

    @classmethod
    def from_checkpoint(cls, path):
        """
        Construct model from saved checkpoint.

        AWX availability is automatically determined based on availability
        of the R-matrix.
        """
        checkpoint = torch.load(path)
        hierarchy = PerLevelHierarchy.from_dict(checkpoint['hierarchy'])
        instance = cls(
            hierarchy,
            checkpoint['config'],
            hierarchy.R is not None
        )
        instance.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        instance.classifier.load_state_dict(
            checkpoint['classifier_state_dict']
        )
        return instance

    def forward_bhcn(self, ids, mask):
        """Forward-propagate input to generate classification."""
        local_outputs = self.classifier(
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
        return local_outputs

    def forward_bhcn_awx(self, ids, mask):
        """Forward-propagate input to generate classification, with AWX."""
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

    def forward(self, ids, mask):
        """Wrap around the two forward propagation routine variants.

        This is only implemented for manual single-example FP convenience.
        Internal training scripts call the corresponding forward functions
        directly, avoiding repetitive conditional branching.
        """
        if self.awx:
            return self.forward_bhcn_awx(ids, mask)
        return self.forward_bhcn(ids, mask)

    def save(self, path, optim):
        """Save model state to disk using PyTorch's pickle facilities."""
        checkpoint = {
            'config': self.config,
            'hierarchy': self.classifier.hierarchy.to_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer': optim
        }
        torch.save(checkpoint, path)

    def load(self, path):
        """Load model state from disk."""
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        return checkpoint['optimizer_state_dict']

    def fit_bhcn(
        self,
        optimizer,
        loss_L_weights,
        train_loader,
        val_loader,
        path=None,
        best_path=None
    ):
        """Train the normal (hierarchical loss) variant of DB-BHCN."""
        criterion = torch.nn.NLLLoss()
        criterion_h = torch.nn.NLLLoss(reduction='none')
        val_loss_min = np.Inf
        lambda_L = self.config['lambda_l']
        lambda_H = self.config['lambda_h']

        # Store validation metrics after each epoch
        val_metrics = np.empty((4, 0), dtype=float)

        for epoch in range(1, self.config['epoch'] + 1):
            train_loss = 0
            self.train()
            print('Epoch {}: Training'.format(epoch))
            for batch_idx, data in enumerate(tqdm(train_loader)):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                targets = data['labels']

                local_outputs = self.forward_bhcn(ids, mask)

                # We have two loss functions:
                # (l)ocal (per-level), and
                # (h)ierarchical.
                loss_l = lambda_L * sum([
                    criterion(
                        local_outputs[level].cpu(),
                        targets[:, level]
                    ) * loss_L_weights[level]
                    for level in range(self.classifier.depth)
                ])

                # Hierarchically penalise less (or don't at all) if the
                # prediction itself is wrong at the child level.
                loss_h_levels = []
                for level in range(self.classifier.depth-1):
                    target_child_indices = torch.unsqueeze(
                        targets[:, level + 1], 1).to(self.device)
                    transformed = local_outputs[level + 1] * -1
                    transformed -= transformed.min(1, keepdim=True)[0]
                    transformed /= transformed.max(1, keepdim=True)[0]
                    loss_factors = 1 - torch.squeeze(
                        transformed.gather(1, target_child_indices), 1)
                    loss_h_levels.append(
                        torch.mean(criterion_h(
                            local_outputs[level],
                            torch.index_select(
                                self.classifier.parent_of[level + 1],
                                0,
                                torch.argmax(
                                    local_outputs[level + 1], dim=1
                                )
                            )
                        ) * loss_factors)
                    )
                loss_h = lambda_H * sum(loss_h_levels)
                loss = loss_l + loss_h

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss = train_loss + (loss.item() - train_loss) / (batch_idx + 1)

            print('Epoch {}: Validating'.format(epoch))
            self.eval()
            val_loss = 0

            val_targets = np.empty((0, self.classifier.depth), dtype=int)
            val_outputs = [
                np.empty(
                    (0, self.classifier.level_sizes[level]), dtype=float
                )
                for level in range(self.classifier.depth)
            ]

            with torch.no_grad():
                for batch_idx, data in enumerate(tqdm(val_loader)):
                    ids = data['ids'].to(self.device, dtype=torch.long)
                    mask = data['mask'].to(self.device, dtype=torch.long)
                    targets = data['labels']

                    local_outputs = self.forward_bhcn(ids, mask)

                    # We have two loss functions:
                    # (l)ocal (per-level), and
                    # (h)ierarchical.
                    loss_l = lambda_L * sum([
                        criterion(
                            local_outputs[level].cpu(),
                            targets[:, level]
                        ) * loss_L_weights[level]
                        for level in range(self.classifier.depth)
                    ])

                    # Hierarchically penalise less (or don't at all) if the
                    # prediction itself is wrong at the child level.
                    loss_h_levels = []
                    for level in range(self.classifier.depth-1):
                        target_child_indices = torch.unsqueeze(
                            targets[:, level + 1], 1).to(self.device)
                        transformed = local_outputs[level + 1] * -1
                        transformed -= transformed.min(1, keepdim=True)[0]
                        transformed /= transformed.max(1, keepdim=True)[0]
                        loss_factors = 1 - torch.squeeze(
                            transformed.gather(1, target_child_indices), 1)
                        loss_h_levels.append(
                            torch.mean(criterion_h(
                                local_outputs[level],
                                torch.index_select(
                                    self.classifier.parent_of[level + 1],
                                    0,
                                    torch.argmax(
                                        local_outputs[level + 1], dim=1
                                    )
                                )
                            ) * loss_factors)
                        )
                    loss_h = lambda_H * sum(loss_h_levels)
                    loss = loss_l + loss_h

                    val_loss = val_loss + (loss.item() - val_loss) / (batch_idx + 1)

                    val_targets = np.concatenate([
                        val_targets, targets.cpu().detach().numpy()
                    ])

                    for i in range(len(val_outputs)):
                        val_outputs[i] = np.concatenate([
                            val_outputs[i],
                            local_outputs[i].cpu().detach().numpy()
                        ])

            val_metrics = np.concatenate([
                val_metrics,
                np.expand_dims(
                    get_metrics(
                        {'outputs': val_outputs, 'targets': val_targets},
                        display='print'),
                    axis=1
                )],
                axis=1
            )

            train_loss = train_loss/len(train_loader)
            val_loss = val_loss/len(val_loader)

            if path is not None and best_path is not None:
                optim = optimizer.state_dict()
                self.save(path, optim)
                if val_loss <= val_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}). Saving best model...'.format(val_loss_min, val_loss))
                    val_loss_min = val_loss
                    self.save(best_path, optim)
            print('Epoch {}: Done\n'.format(epoch))
        return val_metrics

    def test_bhcn(self, loader):
        """Test the AWX variant on a dataset."""
        self.eval()

        all_targets = np.empty((0, self.classifier.depth), dtype=bool)
        all_outputs = [np.empty(
            (0, self.classifier.level_sizes[level]),
            dtype=float
        ) for level in range(self.classifier.depth)]

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(loader)):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                targets = data['labels']

                local_outputs = self.forward_bhcn(ids, mask)

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

    def fit_bhcn_awx(
        self,
        optimizer,
        loss_L_weights,
        train_loader,
        val_loader,
        path=None,
        best_path=None
    ):
        """Train the AWX-equipped variant of DB-BHCN."""
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

        for epoch in range(1, self.config['epoch'] + 1):
            train_loss = 0
            self.train()
            print('Epoch {}: Training'.format(epoch))
            for batch_idx, data in enumerate(tqdm(train_loader)):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                targets = data['labels']
                targets_b = get_hierarchical_one_hot(
                    targets, self.classifier.hierarchy.levels
                ).to(self.device, dtype=torch.float)

                local_outputs, awx_output = self.forward_bhcn_awx(ids, mask)

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
                for batch_idx, data in enumerate(tqdm(val_loader)):
                    ids = data['ids'].to(self.device,
                                         dtype=torch.long)
                    mask = data['mask'].to(self.device,
                                           dtype=torch.long)
                    targets = data['labels']
                    targets_b = get_hierarchical_one_hot(
                        targets, self.classifier.hierarchy.levels
                    ).to(self.device, dtype=torch.float)

                    local_outputs, awx_output = self.forward_bhcn_awx(ids, mask)

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

                    val_loss = val_loss + (loss.item() - val_loss) / (batch_idx + 1)

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
                    get_metrics(
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
                self.save(path, optim)
                if val_loss <= val_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving best model...'.format(val_loss_min, val_loss))
                    val_loss_min = val_loss
                    self.save(best_path, optim)
            print('Epoch {}: Done\n'.format(epoch))
        return val_metrics

    def test_bhcn_awx(self, loader):
        """Test the AWX variant on a dataset."""
        self.eval()

        all_targets = np.empty((0, self.classifier.depth), dtype=bool)
        all_outputs = [np.empty(
            (0, self.classifier.level_sizes[level]),
            dtype=float
        ) for level in range(self.classifier.depth)]

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(loader)):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                targets = data['labels']

                _, awx_output = self.forward_bhcn_awx(ids, mask)

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

    def fit(self, train_loader, val_loader, path=None, best_path=None):
        """Initialise training resources and call the corresponding script.

        Either train_bhcn or train_bhcn_awx will be called, depending on
        whether this model instance was configured with AWX or not.
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

        if self.awx:
            return self.fit_bhcn_awx(
                optimizer,
                loss_L_weights,
                train_loader,
                val_loader,
                path,
                best_path
            )

        return self.fit_bhcn(
            optimizer,
            loss_L_weights,
            train_loader,
            val_loader,
            path,
            best_path
        )

    def test(self, loader):
        """Call the corresponding test script.

        Either fit_bhcn or fit_bhcn_awx will be called, depending on
        whether this model instance was configured with AWX or not.
        """
        if self.awx:
            return self.test_bhcn_awx(loader)
        return self.test_bhcn(loader)

    def export(self, dataset_name, bento=False):
        """Export model to ONNX/Bento."""
        self.eval()
        export_trained(
            self.encoder,
            dataset_name,
            'db_bhcn_awx' if self.awx else 'db_bhcn',
            bento=bento
        )

        # Create dummy input for tracing
        batch_size = 1  # Dummy batch size. When exported, it will be dynamic
        x = torch.randn(batch_size, 768, requires_grad=True).to(
            self.device
        )
        name = '{}_{}'.format(
            'db_bhcn_awx' if self.awx else 'db_bhcn',
            dataset_name
        )
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
        """
        Move this module to specified device.

        This overloads the default PT module's to() method to additionally
        set its internal device variable and moves its submodules.
        """
        super().to(device)
        if device is not None:
            self.encoder.to(device)
            self.classifier.to(device)
            self.device = device
        return self


if __name__ == "__main__":
    pass
