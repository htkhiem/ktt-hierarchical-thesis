import torch
import numpy as np
from tqdm import tqdm

from models import model
from utils.distilbert import get_pretrained
from utils.metric import get_metrics


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
        # of size n x n (same as R). Note that x can be a list of vectors instead of one.
        H = H.expand(len(x), n, n)
        # We'll have to duplicate R to multiply with the entire batch here
        M_batch = self.M.expand(len(x), n, n)
        final_out, _ = torch.max(M_batch*H, dim=2)
        return final_out


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


class DB_AC_HMCNN(model.Model, torch.nn.Module):
    """Wrapper class combining DistilBERT and the adapted C-HMCNN model."""

    def __init__(self, hierarchy, config):
        """Construct module."""
        super(DB_AC_HMCNN, self).__init__()
        self.encoder = get_pretrained().to(config['device'])
        self.classifier = H_MCM_Model(
            768,  # DistilBERT outputs 768 values.
            hierarchy,
            config
        ).to(config['device'])
        self.config = config

    def forward(self, ids, mask):
        """Forward-propagate input to generate classification."""
        return self.classifier(self.encoder(ids, mask)[0][:, 0, :])

    def save(self, path, optim):
        """Save model state to disk using PyTorch's pickle facilities."""
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': optim
        }
        torch.save(checkpoint, path)

    def load(self, path):
        """Load model state from disk."""
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
            resume_from=None
    ):
        """Training script for DistilBERT + Adapted C-HMCNN."""
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
                ids = data['ids'].to(self.config['device'], dtype=torch.long)
                mask = data['mask'].to(self.config['device'], dtype=torch.long)
                targets_b = data['labels_b'].to(
                    self.config['device'], dtype=torch.double
                )

                outputs = self.forward(ids, mask)

                # Notation: H = output stacked, Hbar = hbar stacked
                # MCM = max(M * H, dim=1)
                constr_outputs = self.classifier.mcm(outputs)
                # hbar = y * h
                train_outputs = targets_b * outputs.double()
                # max(M * Hbar, dim = 1)
                train_outputs = self.classifier.mcm(train_outputs)

                # (1-y) + max(M * H, dim = 1) + y * max(M * Hbar, dim = 1)
                train_outputs = (1-targets_b)*constr_outputs.double() + (
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
                    ids = data['ids'].to(self.config['device'],
                                         dtype=torch.long)
                    mask = data['mask'].to(self.config['device'],
                                           dtype=torch.long)
                    targets = data['labels']
                    targets_b = data['labels_b'].to(self.config['device'],
                                                    dtype=torch.double)

                    constrained_outputs = self.forward(ids, mask).double()

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
                        targets.cpu().detach().numpy()])
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
                            get_metrics({
                                'outputs': val_outputs,
                                'targets': val_targets
                            }, display='print'), axis=1
                        )
                    ],
                    axis=1
                )

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
        """Test this model on a dataset."""
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
                ids = data['ids'].to(self.config['device'], dtype=torch.long)
                mask = data['mask'].to(self.config['device'], dtype=torch.long)
                targets = data['labels']

                constrained_outputs = self.forward(ids, mask).double()
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
        """Export model to ONNX/Bento."""
        raise RuntimeError


if __name__ == "__main__":
    pass
