"""Implementation of the DistilBERT+Linear model."""
import os

import torch
import numpy as np
from tqdm import tqdm
import bentoml

from models import model
from utils.hierarchy import PerLevelHierarchy
from utils.distilbert import get_pretrained, export_trained

# Special metrics for leaf-only models.
from sklearn import metrics

import logging


class DropoutLinear(torch.nn.Module):
    """A dropout layer plus a linear layer."""

    def __init__(
        self,
        input_dim,
        hierarchy,
        config
    ):
        """Construct module."""
        super(DropoutLinear, self).__init__()
        self.device = config['device']

        # Dropout
        self.dropout = torch.nn.Dropout(p=config['dropout'])
        self.linear = torch.nn.Linear(input_dim, hierarchy.levels[-1])

        # For exporting
        self.hierarchy = hierarchy

    def forward(self, x):
        """Forward-propagate input to generate classification."""
        return self.linear(self.dropout(x))


class DB_Linear(model.Model, torch.nn.Module):
    """Wrapper class combining DistilBERT with the above module."""

    def __init__(
        self,
        hierarchy,
        config
    ):
        """Construct module."""
        super(DB_Linear, self).__init__()
        self.encoder = get_pretrained()
        self.classifier = DropoutLinear(
            768,
            hierarchy,
            config
        )
        self.output_size = hierarchy.levels[-1]
        self.config = config
        self.device = 'cpu'

    @classmethod
    def from_checkpoint(cls, path):
        """Construct model from saved checkpoint."""
        checkpoint = torch.load(path)
        hierarchy = PerLevelHierarchy.from_dict(checkpoint['hierarchy'])
        instance = cls(hierarchy, checkpoint['config'])
        instance.encoder.load_state_dict(
            checkpoint['encoder_state_dict']
        )
        instance.classifier.load_state_dict(
            checkpoint['classifier_state_dict']
        )
        return instance

    def forward(self, ids, mask):
        """Forward-propagate input to generate classification."""
        return self.classifier(
            self.encoder(
                ids, attention_mask=mask
            )[0][:, 0, :]
        )

    def save(self, path, optim):
        """Save model state to disk using PyTorch's pickle facilities."""
        checkpoint = {
            'config': self.config,
            'hierarchy': self.classifier.hierarchy.to_dict(),
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
        """Training script."""
        # Keep min validation (test set) loss so we can separately back up our
        # best-yet model
        val_loss_min = np.Inf

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([
            {
                'params': self.encoder.parameters(),
                'lr': self.config['encoder_lr']
            },
            {
                'params': self.classifier.parameters(),
                'lr': self.config['classifier_lr']
            }
        ])

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
                targets = data['labels'].to(self.device, dtype=torch.long)

                output = self.forward(ids, mask)

                optimizer.zero_grad()

                loss = criterion(output, targets[:, -1])
                loss.backward()
                optimizer.step()

                train_loss = train_loss + (loss.item() - train_loss) / (
                    batch_idx + 1)

            print('Epoch {}: Validating'.format(epoch))
            self.eval()

            val_targets = np.array([], dtype=float)
            val_outputs = np.empty((0, self.output_size), dtype=float)

            with torch.no_grad():
                for batch_idx, data in tqdm(enumerate(val_loader)):
                    ids = data['ids'].to(self.device,
                                         dtype=torch.long)
                    mask = data['mask'].to(self.device,
                                           dtype=torch.long)
                    targets = data['labels'].to(self.device,
                                                dtype=torch.long)

                    output = self.forward(ids, mask)

                    loss = criterion(output, targets[:, -1])

                    val_targets = np.concatenate([
                        val_targets, targets.cpu().detach().numpy()[:, -1]
                    ])
                    val_outputs = np.concatenate([
                        val_outputs, output.cpu().detach().numpy()
                    ])

                    val_metrics = np.concatenate([
                        val_metrics,
                        np.expand_dims(
                            get_metrics(
                                {
                                    'outputs': val_outputs,
                                    'targets': val_targets
                                },
                                display='print'),
                            axis=1
                        )
                    ], axis=1)

                if path is not None and best_path is not None:
                    optim = optimizer.state_dict()
                    self.save(path, optim)
                    if val_loss <= val_loss_min:
                        print('Validation loss decreased ({:.6f} --> {:.6f}).'
                              'Saving best model...'.format(
                                  val_loss_min, val_loss))
                        val_loss_min = val_loss
                        self.save(best_path, optim)
                print('Epoch {}: Done\n'.format(epoch))
        return val_metrics

    def test(self, loader):
        """Test this model on a dataset."""
        self.eval()

        all_targets = np.array([], dtype=bool)
        all_outputs = np.empty((0, self.output_size), dtype=float)

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(loader)):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                targets = data['labels']

                output = self.forward(ids, mask)

                all_targets = np.concatenate([
                    all_targets, targets.numpy()[:, -1]
                ])
                all_outputs = np.concatenate([
                    all_outputs, output.cpu().detach().numpy()
                ])
        return {
            'targets': all_targets,
            'outputs': all_outputs,
        }

    def export(self, dataset_name, bento=False):
        """Export model to ONNX/Bento."""
        self.eval()
        export_trained(
            self.encoder,
            dataset_name,
            'db_linear',
            bento=bento
        )
        # Create dummy input for tracing
        batch_size = 1  # Dummy batch size. When exported, it will be dynamic
        x = torch.randn(batch_size, 768, requires_grad=True).to(
            self.device
        )
        name = '{}_{}'.format('db_linear', dataset_name)
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
        set its internal device variable.
        """
        super().to(device)
        if device is not None:
            self.device = device
        return self


def get_metrics(test_output, display=None, compute_auprc=False):
    """Compute metrics for leaf-only models like this one."""
    local_outputs = test_output['outputs']
    targets = test_output['targets']
    leaf_size = local_outputs.shape[1]

    def generate_one_hot(idx):
        b = np.zeros(leaf_size, dtype=bool)
        b[idx] = 1
        return b

    # Get predicted class indices at each level
    level_codes = np.argmax(local_outputs, axis=1)

    accuracy = metrics.accuracy_score(targets, level_codes)
    precision = metrics.precision_score(
        targets, level_codes, average='weighted', zero_division=0)

    if display == 'log' or display == 'both':
        logging.info('Leaf level:')
        logging.info("Accuracy: {}".format(accuracy))
        logging.info("Precision: {}".format(precision))

    if display == 'print' or display == 'both':
        print('Leaf level:')
        print("Accuracy: {}".format(accuracy))
        print("Precision: {}".format(precision))

    if compute_auprc:
        binarised_targets = np.array([
            generate_one_hot(idx) for idx in targets
        ])
        rectified_outputs = np.concatenate(
            [local_outputs, np.ones((1, leaf_size))],
            axis=0)
        rectified_targets = np.concatenate(
            [binarised_targets, np.ones((1, leaf_size), dtype=bool)],
            axis=0
        )

        auprc_score = metrics.average_precision_score(
            rectified_targets,
            rectified_outputs
        )
        if display == 'log':
            logging.info('Rectified leaf-level AU(PRC) score: {}'.format(
                auprc_score
            ))
        elif display == 'print':
            print('Rectified leaf-level AU(PRC) score: {}'.format(auprc_score))

        return np.array([accuracy, precision, None, None, auprc_score])
    return np.array([accuracy, precision, None, None])


if __name__ == "__main__":
    pass
