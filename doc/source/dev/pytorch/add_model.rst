.. Dev section - developing with PyTorch.

Implementing a model with PyTorch
===================================================

PyTorch is our framework of choice when it comes to massively-parallelised models such as neural networks.

This article will guide you through the process of implementing a model from scratch using KTT's facilities and PyTorch. It will mainly focus on standout aspects (those that are unique to PyTorch).

PyTorch model module structure
------------------------------

Each PyTorch model module ('module for short') in KTT is a self-contained collection of implemented source code, metadata and configuration files. A module defines its own training, checkpointing and exporting procedures. It might also optionally implement a  BentoML service and configuration files for live inference using the integrated BentoML-powered inference system and monitoring using Prometheus/Grafana.

The general folder tree of a PyTorch model is as follows:


The source implementation itself must subclass the abstract ``Model`` class, like in any other framework. Additionally, PyTorch models with additional submodules (bundled example: DB-BHCN and its AWX submodule, or DistilBERT+Adapted C-HMCNN with its ``MCM`` submodule) must implement a ``to(self, device)`` method similar to PyTorch's namesake method to recursively transfer the entire instance to the specified device.

