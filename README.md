# Dependencies
Just recreate the Anaconda environment using the supplied `environment.yaml` file in `./onnx`. Alternatively, run the following commands:

``` sh
conda install pytorch cudatoolkit=11.3 -c pytorch
conda install scikit-learn pandas dask onnx pathspec tomli nltk mako

# For deployment
# IMPORTANT: Make sure to install nvidia-pyindex first.
# NVIDIA's pip listing is served separately.
pip install nvidia-pyindex
pip install onnxruntime-gpu
pip install nvidia-tensorrt
conda install -c nvidia cudnn cuda-cudart libcufft libcurand libcublas
conda install -c huggingface transformers 

pip install sklearn_hierarchical_classification pyarrow
pip install bentoml --pre
```
