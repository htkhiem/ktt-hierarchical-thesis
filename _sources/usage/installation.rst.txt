.. Installation page.

Installation
===================================================

This guide will help you get the system up and running on a single computer.

Downloading
-----------

KTT is not yet available as a versioned release. However, the Git repository itself is directly usable without further building. Clone the repository using your Git client or the command:

.. code-block:: bash

    git clone https://github.com/htkhiem/ktt-hierarchical-thesis.git ktt
    
The repository will be cloned to a folder titled ``ktt``.

Setting up dependencies
-----------------------
**The easy way:** Use an Anaconda distribution (we recommend ``miniconda3``) to install all dependencies at once using the bundled ``environment.yaml``:

.. code-block:: bash
	
    cd ktt
    conda env create -n ktt -f environment.yaml
    
This will install all dependencies using ``conda install``, or ``pip install`` (with the ``pip`` tool local to the environment).

For subsequent usage of the system, simply activate the environment beforehand in your shell:

.. code-block:: bash
	
    conda activate ktt
    
**The manual way:** Should you prefer to install everything by yourself, please see the list of packages in the `environment.yaml` file.
