import torch
import os
import numpy as np
from tqdm import tqdm

from utils.checkpoint import save_checkpoint
from utils.distilbert import get_pretrained
from utils.metric import get_metrics

class AWX(torch.nn.Module):
    def __init__(
        self,
        config,
        hierarchy, # Relies on existing parent_of information
    ):
        super(AWX, self).__init__()
        # If n <= 0, then use the max-mode. Values less than 0 are meaningless.

        self.n = config['awx_norm']
        self.R = hierarchy.R.transpose(0, 1)
        self.nonlinear = torch.nn.Sigmoid()

    def n_norm(self, x, epsilon=1e-6):
        return torch.pow(
            torch.clamp(
                torch.sum(
                    torch.pow(x, self.n),
                    -1 # dim
                ),
                epsilon,
                1-epsilon
            ),
            1./self.n
        )

    # Assume input is only for leaf level and has not been through sigmoid
    def forward(self, inputs):
        output = self.nonlinear(inputs)
        # Stack/duplicate outputs so we have one copy for every class. Each of these copies
        # will go through n_norm, min or max depending on l.
        output = output.unsqueeze(1)
        output = output.expand(-1, self.R.shape[0], -1)
        # Stack/duplicate R matrix to account for minibatch
        # Stacking on first axis so we don't need a separate unsqueeze call
        R_batch = self.R.expand(inputs.shape[0], -1, -1) # input.shape[0] is minibatch size

        if self.n > 1:
            output = self.n_norm(torch.mul(output, R_batch))
        elif self.n > 0: # that is, n = 1. In this case the modulus is simply sum.
            output = torch.clamp(torch.sum(torch.mul(output, R_batch), 2), max=1-1e-4)
        else:
            output = torch.max(torch.mul(output, R_batch), 2)[0] # Only take values, discard indices
        return output

class DB_BHCN(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hierarchy,
        config,
    ):
        super(DB_BHCN, self).__init__()

        # Back up some parameters for use in forward()
        self.depth = len(hierarchy.levels)
        self.output_dim = len(hierarchy.classes)
        self.level_sizes = hierarchy.levels
        self.level_offsets = hierarchy.level_offsets
        self.parent_of = hierarchy.parent_of
        self.device = config['device']

        # First layer only takes in BERT encodings
        self.fc_layers = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, hierarchy.levels[0])
        ])
        torch.nn.init.xavier_uniform_(self.fc_layers[0].weight)
        self.norms = torch.nn.ModuleList([])
        for i in range(1, self.depth):
            self.fc_layers.extend([
                torch.nn.Linear(input_dim + hierarchy.levels[i-1], hierarchy.levels[i])
            ])
            torch.nn.init.xavier_uniform_(self.fc_layers[i].weight)
            self.norms.extend([torch.nn.LayerNorm(hierarchy.levels[i-1], elementwise_affine=False)])
        # Activation functions
        self.hidden_nonlinear = torch.nn.ReLU() if config['hidden_nonlinear'] == 'relu' else torch.nn.Tanh()
        self.output_nonlinear = torch.nn.LogSoftmax(dim=1)

        # Dropout
        self.dropout = torch.nn.Dropout(p=config['dropout'])

    def forward(self, x):
        # We have |D| of these
        local_outputs = torch.zeros((x.shape[0], self.output_dim)).to(self.device)
        output_l1 = self.fc_layers[0](self.dropout(x))
        local_outputs[:, 0 : self.level_offsets[1]] = self.output_nonlinear(output_l1)

        prev_output = self.hidden_nonlinear(output_l1)
        for i in range(1, self.depth):
            output_li = self.fc_layers[i](torch.cat([self.dropout(self.norms[i-1](prev_output)), x], dim=1))
            local_outputs[:, self.level_offsets[i] : self.level_offsets[i + 1]] = self.output_nonlinear(output_li)
            prev_output = self.hidden_nonlinear(output_li)

        return local_outputs

class DB_BHCN_AWX(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hierarchy,
        config,
    ):
        super(DB_BHCN_AWX, self).__init__()

        # Back up some parameters for use in forward()
        self.depth = len(hierarchy.levels)
        self.output_dim = len(hierarchy.classes)
        self.level_sizes = hierarchy.levels
        self.level_offsets = hierarchy.level_offsets
        self.parent_of = hierarchy.parent_of
        self.device = config['device']

        # First layer only takes in BERT encodings
        self.fc_layers = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, hierarchy.levels[0])
        ])
        torch.nn.init.xavier_uniform_(self.fc_layers[0].weight)
        self.norms = torch.nn.ModuleList([])
        for i in range(1, self.depth):
            self.fc_layers.extend([
                torch.nn.Linear(input_dim + hierarchy.levels[i-1], hierarchy.levels[i])
            ])
            torch.nn.init.xavier_uniform_(self.fc_layers[i].weight)
            self.norms.extend([torch.nn.LayerNorm(hierarchy.levels[i-1], elementwise_affine=False)])
        # Activation functions
        self.hidden_nonlinear = torch.nn.ReLU() if config['hidden_nonlinear'] == 'relu' else torch.nn.Tanh()
        self.output_nonlinear = torch.nn.LogSoftmax(dim=1)

        # AWX layer
        self.awx = AWX(config, hierarchy)

        # Dropout
        self.dropout = torch.nn.Dropout(p=config['dropout'])

    def forward(self, x):
        # We have |D| of these
        local_outputs = torch.zeros((x.shape[0], self.output_dim)).to(self.device)
        output_l1 = self.fc_layers[0](self.dropout(x))
        local_outputs[:, 0 : self.level_offsets[1]] = self.output_nonlinear(output_l1)

        prev_output = output_l1
        for i in range(1, self.depth):
            output_li = self.fc_layers[i](torch.cat([self.dropout(self.norms[i-1](self.hidden_nonlinear(prev_output))), x], dim=1))
            local_outputs[:, self.level_offsets[i] : self.level_offsets[i + 1]] = self.output_nonlinear(output_li)
            prev_output = output_li

        # prev_output now contains the last hidden layer's output. Pass it raw (un-ReLUed) to AWX
        awx_output = self.awx(prev_output)

        return local_outputs, awx_output

def gen_bhcn(config, hierarchy):
    encoder = get_pretrained()
    encoder.to(config['device'])
    classifier = DB_BHCN(
        768, # DistilBERT outputs 768 values.
        hierarchy,
        config
    )
    classifier.to(config['device'])
    return encoder, classifier

def gen_bhcn_awx(config, hierarchy):
    encoder = get_pretrained()
    encoder.to(config['device'])
    classifier = DB_BHCN_AWX(
        768, # DistilBERT outputs 768 values.
        hierarchy,
        config
    )
    classifier.to(config['device'])
    return encoder, classifier

# TRAINING FUNCTIONS
def train_bhcn(config, train_loader, val_loader, gen_model, path=None, best_path=None):
    encoder, classifier = gen_model()

    criterion = torch.nn.NLLLoss()
    criterion_h = torch.nn.NLLLoss(reduction='none')

    optimizer = torch.optim.Adam(
        [
            {'params': encoder.parameters(), 'lr': config['encoder_lr']},
            {'params': classifier.parameters(), 'lr': config['cls_lr']}
        ],
    )

    lambda_L = config['lambda_l']
    lambda_H = config['lambda_h']
    gamma_L = config['gamma_l']

    deviations = np.linspace(-gamma_L, gamma_L, classifier.depth)
    loss_L_weights = [1] * classifier.depth
    loss_L_weights -= deviations

    val_loss_min = np.Inf

    # Store validation metrics after each epoch
    val_metrics = np.empty((4, 0), dtype=float)

    for epoch in range(1, config['epoch'] + 1):
        train_loss = 0
        # Put model into training mode. Note that this call DOES NOT train it yet.
        encoder.train()
        classifier.train()
        print('Epoch {}: Training'.format(epoch))
        for batch_idx, data in enumerate(tqdm(train_loader)):
            ids = data['ids'].to(config['device'], dtype = torch.long)
            mask = data['mask'].to(config['device'], dtype = torch.long)
            targets = data['labels']#.to(device, dtype = torch.long)

            features = encoder(ids, mask)[0][:,0,:]
            local_outputs = classifier(features)
            # Split local outputs
            local_outputs = [ local_outputs[:, classifier.level_offsets[i] : classifier.level_offsets[i+1]] for i in range(classifier.depth)]

            # We have two loss functions: (l)ocal (per-level), and (h)ierarchical.
            loss_l = lambda_L * sum([ criterion(
                local_outputs[level].cpu(),
                targets[:, level]
            ) * loss_L_weights[level] for level in range(classifier.depth) ])

            # Hierarchically penalise less (or don't at all) if the prediction itself is wrong at the child level.
            loss_h_levels = []
            for level in range(classifier.depth-1):
                target_child_indices = torch.unsqueeze(targets[:, level + 1], 1).to(config['device'])
                transformed = local_outputs[level + 1] * -1
                transformed -= transformed.min(1, keepdim=True)[0]
                transformed /= transformed.max(1, keepdim=True)[0]
                loss_factors = 1 - torch.squeeze(transformed.gather(1, target_child_indices), 1)
                loss_h_levels.append(
                    torch.mean(criterion_h(
                        local_outputs[level],
                        torch.index_select(
                            classifier.parent_of[level + 1],
                            0,
                            torch.argmax(local_outputs[level + 1], dim=1)
                        )
                    ) * loss_factors)
                )
            loss_h = lambda_H * sum(loss_h_levels)
            loss = loss_l + loss_h

            # PyTorch defaults to accumulating gradients, but we don't need that here
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + (loss.item() - train_loss) / (batch_idx + 1)

        print('Epoch {}: Validating'.format(epoch))


        # Switch to evaluation (prediction) mode. Again, this doesn't evaluate anything.
        encoder.eval()
        classifier.eval()
        val_loss = 0

        val_targets = np.empty((0, classifier.depth), dtype=int)
        val_outputs = [np.empty((0, classifier.level_sizes[level]), dtype=float) for level in range(classifier.depth)]

        # We're only testing here, so don't run the backward direction (no_grad).
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(val_loader)):
                ids = data['ids'].to(config['device'], dtype = torch.long)
                mask = data['mask'].to(config['device'], dtype = torch.long)
                targets = data['labels']#.to(device, dtype = torch.long)

                features = encoder(ids, mask)[0][:,0,:]
                local_outputs = classifier(features)
                # Split local outputs
                local_outputs = [ local_outputs[:, classifier.level_offsets[i] : classifier.level_offsets[i+1]] for i in range(classifier.depth)]

                # We have two loss functions: (l)ocal (per-level), and (h)ierarchical.
                loss_l = lambda_L * sum([ criterion(
                    local_outputs[level].cpu(),
                    targets[:, level]
                ) * loss_L_weights[level] for level in range(classifier.depth) ])

                # Hierarchically penalise less (or don't at all) if the prediction itself is wrong at the child level.
                loss_h_levels = []
                for level in range(classifier.depth-1):
                    target_child_indices = torch.unsqueeze(targets[:, level + 1], 1).to(config['device'])
                    transformed = local_outputs[level + 1] * -1
                    transformed -= transformed.min(1, keepdim=True)[0]
                    transformed /= transformed.max(1, keepdim=True)[0]
                    loss_factors = 1 - torch.squeeze(transformed.gather(1, target_child_indices), 1)
                    loss_h_levels.append(
                        torch.mean(criterion_h(
                            local_outputs[level],
                            torch.index_select(
                                classifier.parent_of[level + 1],
                                0,
                                torch.argmax(local_outputs[level + 1], dim=1)
                            )
                        ) * loss_factors)
                    )
                loss_h = lambda_H * sum(loss_h_levels)
                loss = loss_l + loss_h

                val_loss = val_loss + (loss.item() - val_loss) / (batch_idx + 1)

                val_targets = np.concatenate([val_targets, targets.cpu().detach().numpy()])

                for i in range(len(val_outputs)):
                    val_outputs[i] = np.concatenate([val_outputs[i], local_outputs[i].cpu().detach().numpy()])

        val_metrics = np.concatenate([val_metrics,
            np.expand_dims(
                get_metrics({'outputs': val_outputs, 'targets': val_targets}, display='print'), axis=1
            )],
            axis=1
        )

        train_loss = train_loss/len(train_loader)
        val_loss = val_loss/len(val_loader)

        if path is not None and best_path is not None:
            # create checkpoint variable and add important data
            checkpoint = {
                'encoder_state_dict': encoder.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
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
def test_bhcn(model, config, loader):
    encoder, classifier = model
    # Switch to evaluation (prediction) mode. Again, this doesn't evaluate anything.
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
            local_outputs = classifier(features)
            # Split local outputs
            local_outputs = [ local_outputs[:, classifier.level_offsets[i] : classifier.level_offsets[i+1]] for i in range(classifier.depth)]

            all_targets = np.concatenate([all_targets, targets])
            for i in range(len(all_outputs)):
                all_outputs[i] = np.concatenate([all_outputs[i], local_outputs[i].cpu().detach().numpy()])

    return {
        'targets': all_targets,
        'outputs': all_outputs,
    }

def train_bhcn_awx(config, train_loader, val_loader, gen_model, path=None, best_path=None):
    encoder, classifier = gen_model()

    criterion_g = torch.nn.BCELoss()
    criterion_l = torch.nn.NLLLoss()

    optimizer = torch.optim.Adam(
        [
            {'params': encoder.parameters(), 'lr': config['encoder_lr']},
            {'params': classifier.parameters(), 'lr': config['cls_lr']}
        ],
    )

    lambda_L = config['lambda_l']
    gamma_L = config['gamma_l']

    deviations = np.linspace(-gamma_L, gamma_L, classifier.depth)
    loss_L_weights = [1] * classifier.depth
    loss_L_weights -= deviations

    val_loss_min = np.Inf

    # Store validation metrics after each epoch
    val_metrics = np.empty((4, 0), dtype=float)

    for epoch in range(1, config['epoch'] + 1):
        train_loss = 0
        # Put model into training mode. Note that this call DOES NOT train it yet.
        encoder.train()
        classifier.train()
        print('Epoch {}: Training'.format(epoch))
        for batch_idx, data in enumerate(tqdm(train_loader)):
            ids = data['ids'].to(config['device'], dtype = torch.long)
            mask = data['mask'].to(config['device'], dtype = torch.long)
            targets = data['labels']#.to(device, dtype = torch.long)
            targets_b = data['labels_b'].to(config['device'], dtype=torch.float)

            features = encoder(ids, mask)[0][:,0,:]
            local_outputs, awx_output = classifier(features)
            # Split local outputs
            local_outputs = [ local_outputs[:, classifier.level_offsets[i] : classifier.level_offsets[i+1]] for i in range(classifier.depth)]

            # We have two loss functions: (l)ocal (per-level), and (g)lobal.
            loss_g = criterion_g(awx_output, targets_b)
            loss_l = lambda_L * sum([ criterion_l(
                local_outputs[level].cpu(),
                targets[:, level]
            ) * loss_L_weights[level] for level in range(classifier.depth) ])

            loss = loss_g + loss_l# + loss_h

            # PyTorch defaults to accumulating gradients, but we don't need that here
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + (loss.item() - train_loss) / (batch_idx + 1)

        print('Epoch {}: Validating'.format(epoch))
        # Switch to evaluation (prediction) mode. Again, this doesn't evaluate anything.
        encoder.eval()
        classifier.eval()
        val_loss = 0

        val_targets = np.empty((0, classifier.depth), dtype=int)
        val_outputs = [np.empty((0, classifier.level_sizes[level]), dtype=float) for level in range(classifier.depth)]

        # We're only testing here, so don't run the backward direction (no_grad).
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(val_loader)):
                ids = data['ids'].to(config['device'], dtype = torch.long)
                mask = data['mask'].to(config['device'], dtype = torch.long)
                targets = data['labels']#.to(device, dtype = torch.long)
                targets_b = data['labels_b'].to(config['device'], dtype = torch.float)

                features = encoder(ids, mask)[0][:,0,:]
                local_outputs, awx_output = classifier(features)
                # Split local outputs
                local_outputs = [ local_outputs[:, classifier.level_offsets[i] : classifier.level_offsets[i+1]] for i in range(classifier.depth)]

                # We have two loss functions: (l)ocal (per-level), and (g)lobal.
                loss_g = criterion_g(awx_output, targets_b)
                loss_l = lambda_L * sum([ criterion_l(
                    local_outputs[level].cpu(),
                    targets[:, level]
                ) * loss_L_weights[level] for level in range(classifier.depth) ])
                loss = loss_g + loss_l

                val_loss = val_loss + (loss.item() - val_loss) / (batch_idx + 1)

                val_targets = np.concatenate([val_targets, targets.cpu().detach().numpy()])

                # Split AWX output into levels
                awx_outputs = [ awx_output[:, classifier.level_offsets[i] : classifier.level_offsets[i + 1]] for i in range(classifier.depth) ]

                for i in range(len(val_outputs)):
                    val_outputs[i] = np.concatenate([val_outputs[i], awx_outputs[i].cpu().detach().numpy()])

        val_metrics = np.concatenate([val_metrics,
            np.expand_dims(
                get_metrics({'outputs': val_outputs, 'targets': val_targets}, display='print'), axis=1
            )],
            axis=1
        )

        train_loss = train_loss/len(train_loader)
        val_loss = val_loss/len(val_loader)

        if path is not None and best_path is not None:
            checkpoint = {
                'encoder_state_dict': encoder.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            best_yet = False
            if val_loss <= val_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving best model...'.format(val_loss_min,val_loss))
                best_yet = True
                val_loss_min = val_loss

            save_checkpoint(
                checkpoint,
                best_yet,
                path,
                best_path
            )
        print('Epoch {}: Done\n'.format(epoch))
    return (encoder, classifier), val_metrics

# Useful for running trained model on test set
def test_bhcn_awx(model, config, loader):
    encoder, classifier = model
    # Switch to evaluation (prediction) mode. Again, this doesn't evaluate anything.
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
            _, awx_output = classifier(features)

            # Cut AWX outputs to levels
            awx_outputs = [ awx_output[:, classifier.level_offsets[i] : classifier.level_offsets[i + 1]] for i in range(classifier.depth) ]

            all_targets = np.concatenate([all_targets, targets])
            for i in range(len(all_outputs)):
                all_outputs[i] = np.concatenate([all_outputs[i], awx_outputs[i].cpu().detach().numpy()])

    return {
        'targets': all_targets,
        'outputs': all_outputs,
    }

if __name__ == "__main__":
    pass
