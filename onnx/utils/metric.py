# This file defines the function used to compute and output metrics from the training process.
# It is capable of printing to the screen and/or a log file.
from sklearn import metrics
import numpy as np
import torch
import logging

# local_outputs: list of Torch tensors, each represeting scores for a hierarchical level.
# targets: list of category codes, ordered in hierarchical order (top to bottom). This can be taken straight from the 'codes' column.
def get_metrics(test_output, display=None, compute_auprc = False):
    local_outputs = test_output['outputs']
    targets = test_output['targets']
    leaf_size = local_outputs[-1].shape[1]
    depth = len(local_outputs)
    def generate_one_hot(idx):
        b = np.zeros(leaf_size, dtype=bool)
        b[idx] = 1
        return b

    # Get predicted class indices at each level
    level_codes = [
        np.argmax(local_outputs[level], axis=1)
        for level in range(len(local_outputs))
    ]

    accuracies = [ metrics.accuracy_score(targets[:, level], level_codes[level]) for level in range(depth) ]
    precisions = [ metrics.precision_score(targets[:, level], level_codes[level], average='weighted', zero_division=0) for level in range(depth) ]

    global_accuracy = sum(accuracies)/len(accuracies)
    global_precision = sum(precisions)/len(precisions)

    if display == 'log' or display == 'both':
        for i in range(depth):
            logging.info('Level {}:'.format(i))
            logging.info("Accuracy: {}".format(accuracies[i]))
            logging.info("Precision: {}".format(precisions[i]))
        logging.info('Global level:')
        logging.info("Accuracy: {}".format(global_accuracy))
        logging.info("Precision: {}".format(global_precision))
    if display == 'print' or display == 'both':
        for i in range(depth):
            print('Level {}:'.format(i))
            print("Accuracy: {}".format(accuracies[i]))
            print("Precision: {}".format(precisions[i]))
        print('Global level:')
        print("Accuracy: {}".format(global_accuracy))
        print("Precision: {}".format(global_precision))

    if compute_auprc:
        binarised_targets = np.array([generate_one_hot(lst[-1]) for lst in targets])
        rectified_outputs = np.concatenate([local_outputs[-1], np.ones((1, local_outputs[-1].shape[1]))], axis=0)
        rectified_targets = np.concatenate([binarised_targets, np.ones((1, leaf_size), dtype=bool)], axis=0)

        auprc_score = metrics.average_precision_score(rectified_targets, rectified_outputs)
        if display == 'log':
            logging.info('Rectified leaf-level AU(PRC) score: {}'.format(auprc_score))
        elif display == 'print':
            print('Rectified leaf-level AU(PRC) score: {}'.format(auprc_score))

        return np.array([accuracies[-1], precisions[-1], global_accuracy, global_precision, auprc_score])
    return np.array([accuracies[-1], precisions[-1], global_accuracy, global_precision])
