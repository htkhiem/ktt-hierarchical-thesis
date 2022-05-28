KTT Hierarchical Classification System
--------------------------------------

KTT Hierarchical Classification System (KTT for short) is a training and inference framework specifically designed for hierarchial multilabel text classification (HMTC) workloads. It is preloaded with seven classification models and a builtin GPU-powered inference pipeline based on the BentoML model inference framework.

# Dependencies
We highly recommend using an Anaconda distribution (such as Miniconda) to install all dependencies at the correct version. Simply use the supplied `environment.yaml` file to create an environment for KTT:

```bash
conda env create -n ktt -f ./environment.yaml
conda activate ktt
```

This is a complete and minimal environment with just the dependencies needed to run KTT to its full capabilities. For building documentation, please refer to that section.

# Documentation

Our documentation is hosted publicly at [our GitHub Page](https://htkhiem.github.io/ktt-hierarchical-thesis/).

Documentation source is located in `onnx/doc`. We use Sphinx with the RTD theme.

## Additional dependencies
Building documentation requires you to install additional dependencies not covered by our `environment.yaml`:

- `sphinx`
- `sphinx-click`
- `sphinxcontrib-bibtex`
- `sphinxcontrib-svg2pdfconverter`
- `sphinx-rtd-theme`

Ensure you are in KTT's conda environment (as doc-building still needs to import KTT's Python modules and thus requires all the runtime dependencies, too) and install these via `pip` as follows:

```bash
# Assuming said environment was named 'ktt'
conda activate ktt-lts
pip install sphinx sphinx-click sphinxcontrib-bibtex sphinx-rtd-theme
```

## Building
```
cd doc
make html
```

## Reading the docs
Open `onnx/doc/build/index.html` with your browser.
