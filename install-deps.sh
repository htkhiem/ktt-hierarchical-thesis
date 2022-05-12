#!/bin/sh
conda install pytorch cudatoolkit=11.3 -c pytorch -y
conda install pandas transformers=4 scikit-learn tqdm nltk -y
conda install -c conda-forge click skl2onnx ray-tune onnxruntime dvc=1 -y
pip install pyarrow evidently bentoml==0.13.1 sklearn_hierarchical_classification HEBO
