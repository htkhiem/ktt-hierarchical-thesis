"""Utilities for exporting models."""
import transformers as tr
import os
import json
import torch.onnx
import bentoml

from models import tfidf_hsgd, db_bhcn, db_ahmcnf, db_achmcnn




# dict mapping classifier names with their init functions
SUPPORTED_CLASSIFIERS = {
    'db_bhcn': db_bhcn.gen_bhcn,
    'db_bhcn_awx': db_bhcn.gen_bhcn_awx,
    'db_ahmcnf': db_ahmcnf.gen,
    'db_achmcnn': db_achmcnn.gen,
    'db_linear': None,
    'tfidf_lsgd': None,
    'tfidf_hsgd': tfidf_hsgd.gen
}


def export_classifier(
        classifier_state_dict,
        classifier_name,
        dataset_name,
        config,
        hierarchy,
):
    # Check if name is one
    assert(classifier_name in SUPPORTED_CLASSIFIERS.keys())
    # Init classifier model
    _, model = SUPPORTED_CLASSIFIERS[classifier_name](config, hierarchy)
    model.load_state_dict(classifier_state_dict)
    model.eval()

    # Create dummy input for tracing
    batch_size = 1  # Dummy batch size. When exported, it will be dynamic
    x = torch.randn(batch_size, 768, requires_grad=True).to(config['device'])

    name = '{}_{}'.format(classifier_name, dataset_name)
    path = 'output/{}/classifier/'.format(name)

    if not os.path.exists(path):
        os.makedirs(path)

    path += 'classifier.onnx'

    # Clear previous versions
    if os.path.exists(path):
        os.remove(path)

    # Export into transformers model .bin format
    torch.onnx.export(
        model,                      # model being run
        x,                          # model input (or a tuple for multiple inputs)
        path,                       # where to save the model (can be a file or file-like object)
        export_params=True,         # store the trained parameter weights inside the model file
        opset_version=11,           # the ONNX version to export the model to
        do_constant_folding=True,   # whether to execute constant folding for optimization
        input_names=['input'],    # the model's input names
        output_names=['output'],  # the model's output names
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Set first dim of in/out as dynamic (variable)
    )

    # Optionally save to BentoML model store. Pack hierarchical metadata along with model for convenience.
    if config['bento']:
        bentoml.onnx.save('classifier_' + name, path, metadata=hierarchy_json)


if __name__ == 'main':
    pass
