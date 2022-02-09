import torch
import os
import numpy as np
from tqdm import tqdm

from utils.checkpoint import save_checkpoint
from utils.distilbert import get_pretrained
from utils.metric import get_metrics

from tqdm import tqdm

class MCM(torch.nn.Module):
    def __init__(self, M):
        super(MCM, self).__init__()
        self.M = M

    def forward(self, x):
        n = self.M.shape[1]
        H = x.unsqueeze(1) # Add a new dimension
        # Duplicate x along the new dimension to create a list of 2D matrices
        # of size n x n (same as R). Note that x can be a list of vectors instead of one.
        H = H.expand(len(x), n, n)
        # We'll have to duplicate R to multiply with the entire batch here
        M_batch = self.M.expand(len(x), n, n)
        final_out, _ = torch.max(M_batch*H, dim = 2)
        return final_out

class H_MCM_Model(torch.nn.Module):
    def __init__(self, input_dim, hierarchy, config):
        super(H_MCM_Model, self).__init__()

        self.depth = len(hierarchy.levels)
        self.level_sizes = hierarchy.levels
        self.level_offsets = hierarchy.level_offsets
        self.layer_count = config['h_layer_count']
        self.mcm = MCM(hierarchy.M)

        fc = []
        if self.layer_count == 1:
            fc.append(torch.nn.Linear(input_dim, output_dim))
        else:
            for i in range(self.layer_count):
                if i == 0:
                    fc.append(torch.nn.Linear(input_dim, config['h_hidden_dim']))
                elif i == self.layer_count - 1:
                    fc.append(torch.nn.Linear(config['h_hidden_dim'], len(hierarchy.classes)))
                else:
                    fc.append(torch.nn.Linear(config['h_hidden_dim'], config['h_hidden_dim']))

        self.fc = torch.nn.ModuleList(fc)
        self.drop = torch.nn.Dropout(config['h_dropout'])
        self.sigmoid = torch.nn.Sigmoid()
        if config['h_nonlinear'] == 'tanh':
            self.f = torch.nn.Tanh()
        else:
            self.f = torch.nn.ReLU()

    def forward(self, x):
        for i in range(self.layer_count):
            if i == self.layer_count - 1:
                x = self.sigmoid(self.fc[i](x))
            else:
                x = self.f(self.fc[i](x))
                x = self.drop(x)
        if self.training:
            return x
        return self.mcm(x)

def gen(config, hierarchy):
    encoder = get_pretrained()
    encoder.to(config['device'])
    depth = len(hierarchy.levels)

    classifier = H_MCM_Model(
        768, # DistilBERT outputs 768 values.
        hierarchy,
        config
    )
    classifier.to(config['device'])
    return encoder, classifier

def train(config, train_loader, val_loader, gen_model, path=None, best_path=None):
    encoder, classifier = gen_model()

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        [
            {'params': encoder.parameters(), 'lr': config['encoder_lr'],},
            {'params': classifier.parameters(), 'lr': config['classifier_lr']}
        ],
    )
    val_loss_min = np.Inf
    # Store validation metrics after each epoch
    val_metrics = np.empty((4, 0), dtype=float)
    for epoch in range(1, config['epoch'] + 1):
        train_loss = 0
        val_loss = 0
        encoder.train()
        classifier.train()
        print('Epoch {}: Training'.format(epoch))
        for batch_idx, data in enumerate(tqdm(train_loader)):
            ids = data['ids'].to(config['device'], dtype = torch.long)
            mask = data['mask'].to(config['device'], dtype = torch.long)
            targets_b = data['labels_b'].to(config['device'], dtype = torch.double)

            features = encoder(ids, mask)[0][:,0,:]
            outputs = classifier(features)

            # Notation: H = output stacked, Hbar = hbar stacked
            constr_outputs = classifier.mcm(outputs) # MCM = max(M * H, dim=1)
            train_outputs = targets_b * outputs.double() # hbar = y * h
            train_outputs = classifier.mcm(train_outputs) # max(M * Hbar, dim = 1)

            # (1-y) + max(M * H, dim = 1) + y * max(M * Hbar, dim = 1) versus y
            train_outputs = (1-targets_b)*constr_outputs.double() + targets_b*train_outputs
            loss = criterion(train_outputs, targets_b)

            predicted = constr_outputs.data > 0.5

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))

        print('Epoch {}: Validating'.format(epoch))
        encoder.eval()
        classifier.eval()

        val_targets = np.empty((0, classifier.depth), dtype=int)
        val_outputs = [np.empty((0, classifier.level_sizes[level]), dtype=float) for level in range(classifier.depth)]
        # We're only testing here, so don't run the backward direction (no_grad).
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(val_loader)):
                ids = data['ids'].to(config['device'], dtype = torch.long)
                mask = data['mask'].to(config['device'], dtype = torch.long)
                targets = data['labels']
                targets_b = data['labels_b'].to(config['device'], dtype = torch.double)

                features = encoder(ids, mask)[0][:,0,:]
                constrained_outputs = classifier(features).double()

                loss = criterion(constrained_outputs, targets_b)

                # Split local outputs
                local_outputs = [ constrained_outputs[:, classifier.level_offsets[i] : classifier.level_offsets[i+1]] for i in range(classifier.depth)]

                val_loss = val_loss + ((1 / (batch_idx + 1)) * (loss.item() - val_loss))

                val_targets = np.concatenate([val_targets, targets.cpu().detach().numpy()])
                for i in range(len(val_outputs)):
                    val_outputs[i] = np.concatenate([val_outputs[i], local_outputs[i].cpu().detach().numpy()])

            train_loss = train_loss/len(train_loader)
            val_loss = val_loss/len(val_loader)

            val_metrics = np.concatenate(
                [
                    val_metrics,
                    np.expand_dims(
                        get_metrics({'outputs': val_outputs, 'targets': val_targets}, display='print'), axis=1
                    )
                ],
                axis=1
            )

            if path is not None and best_path is not None:
                # create checkpoint variable and add important data
                checkpoint = {
                    'encoder_state_dict': encoder.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                # save checkpoint
                best_yet = False
                if val_loss <= val_loss_min:
                    best_yet = True
                    print('Validation loss decreased ({:.6f} --> {:.6f}). Saving best model...'.format(val_loss_min,val_loss))
                    # save checkpoint as best model
                    val_loss_min = val_loss
                save_checkpoint(checkpoint, best_yet, path, best_path)
        print('Epoch {}: Done\n'.format(epoch))
    return (encoder, classifier), val_metrics

# Alternative: just load from disk
def test(model, config, loader):
    encoder, classifier = model
    encoder.eval()
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
            constrained_outputs = classifier(features).double()
            # Split local outputs
            local_outputs = [ constrained_outputs[:, classifier.level_offsets[i] : classifier.level_offsets[i+1]] for i in range(classifier.depth)]

            all_targets = np.concatenate([all_targets, targets])
            for i in range(len(all_outputs)):
                all_outputs[i] = np.concatenate([all_outputs[i], local_outputs[i].cpu().detach().numpy()])

    return {
        'targets': all_targets,
        'outputs': all_outputs,
    }

if __name__ == "__main__":
    pass
