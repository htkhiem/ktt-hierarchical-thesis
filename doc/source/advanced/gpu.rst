.. GPU-accelerated inference guide

Inferencing with GPUs
===================================================

By default, we support accelerating training using CUDA (on supported NVIDIA GPUs). However, you may also use CUDA to accelerate the *inference* workload and achieve much higher API throughput than what's possible using a CPU, especially for DistilBERT-based models.

Prerequisites
-------------

- Currently, GPU inference has only been tested on Linux hosts. Specifically, we require a Linux installation on an ``x86_64`` architecture with kernel version of at least ``3.10``. To check your currentn kernel version, run ``uname -a``.
- NVIDIA drivers and CUDA ``>11.3``. Please use the official (proprietary) drivers instead of the open-source ``nouveau`` one.
  - If you want to use Docker containerisation, then Docker ``19.03`` or newer is required.
  - If your BentoService has monitoring capabilities enabled (i.e. it runs as a ``docker-compose`` network) then you also need to install the NVIDIA Container Runtime. Arch Linux users can install ``nvidia-container-runtime`` from the AUR.

GPU-based inference using Bentos
--------------------------------

If you decide to stop at raw BentoServices for deployment to your inference server, you simply need to ensure that the server has direct access to a compatible NVIDIA GPU and install all drivers and dependencies accordingly. KTT-generated BentoServices are designed to automatically take advantage of the *first*-found NVIDIA GPU.

To verify that the GPU is being used, you should use ``nvidia-smi``. If the Bento is using your GPU:

- Video memory utilisation will be higher than idle. For DistilBERT-based models, expect usage of around 4GB.
- There is a ``python`` process listed in the process list.

.. code-block:: bash

    > nvidia-smi
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 465.31       Driver Version: 465.31       CUDA Version: 11.3     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
    | N/A   49C    P8     6W /  N/A |    753MiB /  6078MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A    179346      C   /opt/conda/bin/python             745MiB |
    +-----------------------------------------------------------------------------+
    
GPU-based inference for Dockerised services
-------------------------------------------

Docker containers can simply be deployed to a server without needing to care about dependencies. One only needs to ensure that the host machine itself satisfies the hardware requirements and has the correct drivers installed.

Without monitoring capabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Services built without monitoring capabilities are Dockerised into single Docker images. They can be run directly from the terminal using your typical ``docker run`` command.
Due to a recent ``systemd`` architectural redesign, we need a workaround to grant hardware access to the container.

Instead of starting the Docker container as usual, please add ``--gpu all`` and specific ``--device`` options as follows:

.. code-block:: bash

    > docker run --gpus all --device /dev/nvidia0 --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset --device /dev/nvidiactl <your arguments>

In ``<your arguments>``, fill in your usual Docker arguments such as the image ID and port forwarding. Again, you can verify that the GPU is being utilised by running ``nvidia-smi`` and looking out for the signs described above. Of course, you should run it on the host machine's terminal instead of within the shell of the container.

.. note::

   Due to how BentoML GPU-enabled base images are configured, you might encounter errors like the following:

   .. code-block::

      RuntimeError: Click will abort further execution because Python was configured to use ASCII as encoding for the environment. Consult https://click.palletsprojects.com/unicode-support/ for mitigation steps.

      This system supports the C.UTF-8 locale which is recommended. You might be able to resolve your issue by exporting the following environment variables:

      export LC_ALL=C.UTF-8
      export LANG=C.UTF-8


   In other words, the base images were configured with their system locale being set to ASCII, which is potentially buggy for Python interpreters, which can accept Unicode too. The fix is to simply do as they said - override the base image's system locale to UTF-8. One way to do that would be through the ``docker run`` command itself by adding ``-e LC_ALL='C.UTF-8' -e LANG='C.UTF-8'`` to the arguments list.

With monitoring capabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Services built with monitoring capabilities contain not just the inference server itself but also several other services, namely the Prometheus time series database, Grafana dashboard and an Evidently-based model metrics app. All of these are supposed to be run together as a ``docker-compose`` network instead of being separately and manually started. Our ``docker-compose.yaml`` configuration already includes all the workarounds, so you only need to ensure the host system's NVIDIA GPU is functional and accessible.
