import onnxruntime
import argparse, json
import numpy as np
from transformers import DistilBertTokenizerFast

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='The name of the dataset your desired model was trained against. By default we\'ll look for a model trained against the supplied Walmart_30k.')
    parser.add_argument('-m', '--model', help="""The name of the model to use. This will be combined with the specified dataset to load the correct trained model. Available models:
\tdb_bhcn\t\t(DistilBERT Branching Hierarchical Classifier)
\tdb_bhcn_awx\t\t(DistilBERT Branching Hierarchical Classifier + Adjacency Wrapping Matrix)
\tdb_ahmcnf\t\t(Adapted HMCN-F model running on DistilBERT encodings)
\tdb_achmcnn\t\t(Adapted C-HMCNN model running on DistilBERT encodings)
\ttfidf_hsgd\t\t(Internal-node SGD classifier hierarchy using tf-idf encodings)
By default, all models are run.""")
    parser.add_argument('-v', '--verbose', action='store_true', help='Print more information to the console (for debugging purposes).')
    parser.add_argument('-c', '--cpu', action='store_true', help='Only run on CPU. Use this if you have to run without CUDA support (warning: depressingly slow).')

    args = parser.parse_args()

    verbose = args.verbose if args.verbose else False
    dataset_name = args.dataset if args.dataset else 'Walmart_30k'
    model_name = args.model if args.model else 'db_bhcn'

    folder_name = 'output/{}_{}'.format(model_name, dataset_name)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    encoder_session = onnxruntime.InferenceSession(folder_name + "/encoder/model.onnx")
    classifier_session = onnxruntime.InferenceSession(folder_name + "/classifier/classifier.onnx")

    inputs = tokenizer('Apple iPhone 6S 64GB smartphone iOS', return_tensors="np")
    encoder_outputs = encoder_session.run(output_names=["last_hidden_state"], input_feed=dict(inputs))
    print('DistilBERT ONNX output shape:', encoder_outputs[0][:, 0, :].shape) # expect (1, 768)
    classifier_outputs = classifier_session.run(None, {'input': encoder_outputs[0][:, 0, :]})
    print('Classifier output:', classifier_outputs)
    print('Classifier output shape:', classifier_outputs[0].shape) # for Walmart_30k, expect (1, 324)

    scores = classifier_outputs[0]

    # Segmented argmax, as usual
    with open(folder_name + '/hierarchy.json', 'r') as infile:
        hierarchy = json.load(infile)
        level_offsets = hierarchy['level_offsets']
        level_sizes = hierarchy['level_sizes']
        classes = hierarchy['classes']

    pred_codes = [
        int(np.argmax(scores[:, level_offsets[level] : level_offsets[level + 1]], axis=1) + level_offsets[level])
        for level in range(len(level_sizes))
    ]

    print('Predicted codes:', pred_codes)

    pred_names = [
        classes[i] for i in pred_codes
    ]

    print(pred_names)
