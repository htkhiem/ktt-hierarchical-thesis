import torch
import os
import numpy as np
from tqdm import tqdm

from utils.checkpoint import save_checkpoint
from utils.distilbert import get_pretrained
from utils.metric import get_metrics

from tqdm import tqdm

class HMCNF(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hierarchy,
        config
    ):
        super(HMCNF, self).__init__()

        # Back up some parameters for use in forward()
        self.depth = len(hierarchy.levels)
        self.global_weight = config['global_weight']
        self.output_dim = len(hierarchy.classes)
        self.level_sizes = hierarchy.levels
        self.level_offsets = hierarchy.level_offsets
        self.parent_of = hierarchy.parent_of
        self.device = config['device']

        # Construct global layers (main flow)
        global_layers = []
        global_layer_norms = []
        for i in range(self.depth):
            if i == 0:
                global_layers.append(
                    torch.nn.Linear(input_dim, config['global_hidden_sizes'][0]))
            else:
                global_layers.append(
                    torch.nn.Linear(config['global_hidden_sizes'][i-1] + input_dim, config['global_hidden_sizes'][i]))
            global_layer_norms.append(torch.nn.LayerNorm(config['global_hidden_sizes'][i]))
        self.global_layers = torch.nn.ModuleList(global_layers)
        self.global_layer_norms = torch.nn.ModuleList(global_layer_norms)
        # Global prediction layer
        self.global_prediction_layer = torch.nn.Linear(
            config['global_hidden_sizes'][-1] + input_dim,
            len(hierarchy.classes)
        )

        # Construct local branches (local flow).
        # Each local branch has two linear layers: a transition layer and a local
        # classification layer
        transition_layers = []
        local_layer_norms = []
        local_layers = []

        for i in range(self.depth):
            transition_layers.append(
                torch.nn.Linear(config['global_hidden_sizes'][i], config['local_hidden_sizes'][i]),
            )
            local_layer_norms.append(
                torch.nn.LayerNorm(config['local_hidden_sizes'][i])
            )
            local_layers.append(
                torch.nn.Linear(config['local_hidden_sizes'][i], hierarchy.levels[i])
            )
            self.local_layer_norms = torch.nn.ModuleList(local_layer_norms)
            self.transition_layers = torch.nn.ModuleList(transition_layers)
            self.local_layers = torch.nn.ModuleList(local_layers)

        # Activation functions
        self.hidden_nonlinear = torch.nn.ReLU() if config['hidden_nonlinear'] == 'relu' else torch.nn.Tanh()
        self.output_nonlinear = torch.nn.Sigmoid()

        # Dropout
        self.dropout = torch.nn.Dropout(p=config['dropout'])

    def forward(self, x):
        # We have |D| hidden layers plus one global prediction layer
        local_outputs = torch.zeros((x.shape[0], self.output_dim)).to(self.device)
        output = x # Would be global path output until the last step
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
                )
            )

            # Local path. Note the dropout between the transition ReLU layer and the local layer.
            local_output = self.dropout(
                self.hidden_nonlinear(
                    self.local_layer_norms[i](self.transition_layers[i](output))
                )
            )
            local_output = self.output_nonlinear(self.local_layers[i](local_output))
            local_outputs[:, self.level_offsets[i] : self.level_offsets[i + 1]] = local_output

            # Dropout main flow for next layer
            output = self.dropout(output)

        global_outputs = self.output_nonlinear(
            self.global_prediction_layer(torch.cat([output, x], dim=1))
        )
        output = self.global_weight * global_outputs + (1 - self.global_weight) * local_outputs
        return output, local_outputs

def gen(config, hierarchy):
    encoder = get_pretrained()
    encoder.to(config['device'])

    classifier = HMCNF(
        768, # DistilBERT outputs 768 values.
        hierarchy,
        config
    )
    classifier.to(config['device'])

    return encoder, classifier

def train(config, train_loader, val_loader, gen_model, path=None, best_path=None):
    encoder, classifier = gen_model()

    # HMCN-F's implementation uses a global-space vector of parent-class indices.
    global_parent_of = torch.cat(classifier.parent_of, axis=0).to(config['device'])

    # Keep min validation (test set) loss so we can separately back up our best-yet model
    val_loss_min = np.Inf

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=classifier.parameters(), lr=config['cls_lr'])

    # Store validation metrics after each epoch
    val_metrics = np.empty((4, 0), dtype=float)

    # Hierarchical loss gain
    lambda_h = config['lambda_h']
    for epoch in range(1, config['epoch'] + 1):
        train_loss = 0
        val_loss = 0
        # Put model into training mode. Note that this call DOES NOT train it yet.
        classifier.train()
        print('Epoch {}: Training'.format(epoch))
        for batch_idx, data in enumerate(tqdm(train_loader)):
            ids = data['ids'].to(config['device'], dtype = torch.long)
            mask = data['mask'].to(config['device'], dtype = torch.long)
            targets = data['labels'].to(config['device'], dtype = torch.float)
            targets = data['labels']
            targets_b = data['labels_b'].to(config['device'], dtype = torch.float)

            features = encoder(ids, mask)[0][:,0,:]
            output, local_outputs = classifier(features)
            # Split local outputs
            local_outputs = [ local_outputs[:, classifier.level_offsets[i] : classifier.level_offsets[i+1]] for i in range(classifier.depth)]

            optimizer.zero_grad()

            # We have three loss functions: (g)lobal, (l)ocal, and (h)ierarchical.
            loss_g = criterion(output, targets_b)
            loss_l = sum([ criterion(
                local_outputs[level],
                targets_b[:, classifier.level_offsets[level] : classifier.level_offsets[level + 1]]
            ) for level in range(classifier.depth)])
            # output_cpu = output.cpu().detach()
            loss_h = torch.sum(lambda_h * torch.clamp(
                output -
                output.index_select(1, global_parent_of),
                min=0) ** 2)
            loss = loss_g + loss_l + loss_h

            # PyTorch defaults to accumulating gradients, but we don't need that here
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + (loss.item() - train_loss) / (batch_idx + 1)

        print('Epoch {}: Testing'.format(epoch))
        # Switch to evaluation (prediction) mode. Again, this doesn't evaluate anything.
        classifier.eval()

        val_targets = np.empty((0, classifier.depth), dtype=int)
        val_outputs = [np.empty((0, classifier.level_sizes[level]), dtype=float) for level in range(classifier.depth)]

        # We're only testing here, so don't run the backward direction (no_grad).
        with torch.no_grad():
            for batch_idx, data in tqdm(enumerate(val_loader)):
                ids = data['ids'].to(config['device'], dtype = torch.long)
                mask = data['mask'].to(config['device'], dtype = torch.long)
                targets = data['labels']
                targets_b = data['labels_b'].to(config['device'], dtype = torch.float)

                features = encoder(ids, mask)[0][:,0,:]
                output, local_outputs = classifier(features)
                # Split local outputs
                local_outputs = [ local_outputs[:, classifier.level_offsets[i] : classifier.level_offsets[i+1]] for i in range(classifier.depth)]

                loss_g = criterion(output, targets_b)
                loss_l = sum([ criterion(
                    local_outputs[level],
                    targets_b[:, classifier.level_offsets[level] : classifier.level_offsets[level + 1]]
                ) for level in range(classifier.depth)])
                # output_cpu = output.cpu().detach()
                loss_h = torch.sum(lambda_h * torch.clamp(
                    output -
                    output.index_select(1, global_parent_of),
                    min=0) ** 2)
                loss = loss_g + loss_l + loss_h

                val_loss = val_loss + (loss.item() - val_loss) / (batch_idx + 1)

                val_targets = np.concatenate([val_targets, targets.cpu().detach().numpy()])
                for i in range(len(val_outputs)):
                    val_outputs[i] = np.concatenate([val_outputs[i], local_outputs[i].cpu().detach().numpy()])

            val_metrics = np.concatenate(
                [
                    val_metrics,
                    np.expand_dims(
                        get_metrics({'outputs': val_outputs, 'targets': val_targets}, display='print'), axis=1
                    )
                ],
                axis=1
            )
            train_loss = train_loss/len(train_loader)
            val_loss = val_loss/len(val_loader)

            if path is not None and best_path is not None:
            # create checkpoint variable and add important data
                checkpoint = {
                    'state_dict': classifier.state_dict(),
                    'optimizer': optimizer.state_dict()
                }

                best_yet = False
                if val_loss <= val_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}). Saving best model...'.format(val_loss_min,val_loss))
                    # save checkpoint as best model
                    best_yet = True
                    val_loss_min = val_loss
                save_checkpoint(checkpoint, best_yet, path, best_path)
            print('Epoch {}: Done\n'.format(epoch))
    return (encoder, classifier), val_metrics

# Alternative: just load from disk
def test(model, config, loader):
    encoder, classifier = model
    # Switch to evaluation (prediction) mode. Again, this doesn't evaluate anything.
    classifier.eval()

    all_targets = np.empty((0, classifier.depth), dtype=bool)
    all_outputs = [np.empty((0, classifier.level_sizes[level]), dtype=float) for level in range(classifier.depth)]

    # We're only testing here, so don't run the backward direction (no_grad).
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader)):
            ids = data['ids'].to(config['device'], dtype = torch.long)
            mask = data['mask'].to(config['device'], dtype = torch.long)
            targets = data['labels']

            features = encoder(ids, mask)[0][:,0,:]
            _, local_outputs = classifier(features)
            # Split local outputs
            local_outputs = [ local_outputs[:, classifier.level_offsets[i] : classifier.level_offsets[i+1]] for i in range(classifier.depth)]

            all_targets = np.concatenate([all_targets, targets])
            for i in range(len(all_outputs)):
                all_outputs[i] = np.concatenate([all_outputs[i], local_outputs[i].cpu().detach().numpy()])
    return {
        'targets': all_targets,
        'outputs': all_outputs,
    }

if __name__ == "__main__":
    pass
