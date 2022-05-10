KTT Hierarchical Classification System
--------------------------------------

KTT Hierarchical Classification System (KTT for short) is a training and inference framework specifically designed for hierarchial multilabel text classification (HMTC) workloads. It is preloaded with seven classification models and a builtin GPU-powered inference pipeline based on the BentoML model inference framework.

# Dependencies
We highly recommend using an Anaconda distribution (such as Miniconda) to install all dependencies at the correct version. Simply use the supplied `environment.yaml` file to create an environment for KTT:

```bash
conda env create -n ktt -f ./environment.yaml
conda activate ktt
```

# Documentation
Documentation regarding the production training and inference system is located in `onnx/doc`. We use Sphinx with the RTD theme.

## Building
```
cd doc
make html
```

## Reading the docs
Open `onnx/doc/build/index.html` with your browser.
