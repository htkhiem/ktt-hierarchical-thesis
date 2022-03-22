# Dependencies
Just recreate the Anaconda environment using the supplied `environment.yaml` file in `./onnx`. Alternatively, run the following commands:

``` sh
conda install pytorch cudatoolkit=11.3 -c pytorch
conda install scikit-learn pandas dask onnx pathspec tomli nltk mako sphinx sphinx_rtd_theme

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

# Documentation
Documentation regarding the production training and inference system is located in `onnx/doc`. We use Sphinx with the RTD theme. Dependencies for these are already included in the prior listing.

## Building
```
cd onnx/doc
make html
```

## Reading the docs
Open `onnx/doc/build/index.html` with your browser.