.. GPU-accelerated inference guide

Inferencing with GPUs
===================================================

By default, we support accelerating training using CUDA (on supported NVIDIA GPUs). However, you may also use CUDA to accelerate the *inference* workload and achieve much higher API throughput than what's possible using a CPU.

Prerequisites
-------------

- Currently, GPU inference has only been tested on Linux hosts. Specifically, we require a Linux installation on an ``x86_64`` architecture with kernel version of at least ``3.10``. To check your currentn kernel version, run ``uname -a``.
- If you want to use Docker containerisation, then Docker ``19.03`` or newer is required.
- NVIDIA drivers and CUDA ``>11.3``. Please use the official (proprietary) drivers instead of the open-source ``nouveau`` one.

ALl other GPU-related dependencies should already be installed to your ``anaconda`` environment if you installed via the ``onnx/environment.yaml`` file.

GPU-based inference using Bentos
--------------------------------

If you decide to stop at Bentos for deployment to your inference server, you simply need to ensure that the server has direct access to a compatible NVIDIA GPU and install all drivers and dependencies accordingly. Our Bentos are designed to automatically take advantage of the *first*-found NVIDIA GPU.

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
    
GPU-based inference on Docker containers
----------------------------------------

Docker containers can simply be deployed to a server without needing to care about dependencies. One only needs to ensure that the host machine itself satisfies the hardware requirements and has the correct drivers installed. However, due to a recent ``systemd`` architectural redesign, we need a workaround to grant hardware access to the container.

Instead of starting the Docker container as usual, please use this:

.. code-block:: bash

    > docker run --gpus all --device /dev/nvidia0 --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset --device /dev/nvidiactl <your arguments>

In ``<your arguments>``, fill in your usual Docker arguments such as the image ID. Again, you can verify that the GPU is being utilised by running ``nvidia-smi`` and looking out for the signs described above. Of course, you should run it on the host machine's terminal instead of within the shell of the container.
