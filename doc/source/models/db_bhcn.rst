.. DB-BHCN and DB-BHCN+AWX documentation.

DB-BHCN and DB-BHCN+AWX
=======================

API
---

.. autoclass:: models.DB_BHCN
   :members:
   :special-members:

Configuration schema
--------------------
The configuration for this model defines the following hyperparameters:

* ``encoder_lr``: Encoder (DistilBERT) learning rate.
* ``classifier_lr``: Classifier learning rate.
* ``dropout``: The model's dropout rate.
* ``hidden_nonlinear``: The FCFF model's nonlinear type, which can be either ``relu`` or ``tanhh``.
* ``awx_norm``: represents the max that the matrix awx stretches (default 5) (Adjacency Wrapping Matrix layer option).

Checkpoint schema
-----------------
* ``config``: A copy of the configuration dictionary passed to this instance's constructor, either explicitly, or by ``from_checkpoint`` (extracted from a prior checkpoint).
* ``hierarchy``: A serialised dictionary of hierarchical metadata created by ``PerLevelHierarchy.to_dict()``.
* ``encoder_state_dict``: Weights of the DistilBERT model.
* ``classifier_state_dict``: Weights of the classifier.
* ``optimizer_state_dict``: Saved state of the optimiser that was used to train the model for that checkpoint.

Theory 
------
DB-BHCN
~~~~~~~~~~~~~~~~~~~~~~~~~
DB-BHCN stands for the the DistilBERT Branching Hierarchical Classification Network (DB-BHCN) model, which is designed specifically to maximise the utilisation of advanced language comprehension capabilities made available by DistilBERT model.

An example computation graph for the classifier is given below.

.. image:: bhcn.svg
   :width: 320
   :alt: Topology of the BHCN.

The topology's depth matches that of the hierarchy with no additional layer, while its output size at each layer corresponds to the number of classes at the corresponding hierarchical level. Outputs are taken in a residual manner from each layer, 'branching' from the main flow, hence the name. Between layers, the hidden output first goes through a hidden nonlinear function, then a layer normalisation function, then a dropout stage whose probability is a hyperparameter.

Formally, let :math:`x \in \mathbb{R}^{768 \times 1}` be the feature vector, that is, the direct input that came from the fine-tuned DistilBERT instance that goes into the fully-connected flow; :math:`H` be the depth-padded hierarchy object for which :math:`|H|` represents its depth, :math:`C_H` be the set of classes in H where :math:`C_{H_h}` represents the set of classes at level :math:`h`; :math:`W` and :math:`b` be the set of weights and biases where :math:`W_h \in \mathbb{R}^{|C_{H_h}| \times |A^{h-1}|}` and :math:`b_h \in \mathbb{R}^{|A^{h-1}| \times 1}` are the weights and biases of layer :math:`h`, respectively; :math:`z^h \in \mathbb{R}^{|C_{H_h}| \times 1}` represents the linear output from layer :math:`h`, :math:`A^h = \phi(z^h)` being the hidden nonlinear output with :math:`\phi` being a hidden nonlinear activation function (such as ReLU or tanh); and :math:`P^h = \psi(z^h)` being the output scores for classes of level :math:`h` in hierarchy :math:`H`, with :math:`\psi` being the output nonlinear activation function, which in this case is the typical LogSoftmax function.


DB-BHCN's data flow is relatively straightforward. The input text is first encoded by a fine-tuned DistilBERT instance into :math:`x`. This enters the first fully-connected level, which produces :math:`z_1`, which is then transformed by :math:`\phi` and :math:`\psi` into :math:`A_1` and :math:`P_1`, respectively:

.. math::
      z^1 = W^1x + b^1 
.. math::
      A^1 = \phi(z^1) 
.. math::
      P^1 = \psi(z^1) 

Then, for each non-leaf layer :math:`0 < h \neq |H|`, the same happens, with a small difference from the input layer in that the previous output is also concatenated with the original feature vector:

.. math::
   z^h = W^h(x ++ A^{h - 1}) + b^{h} 
.. math::
   A^h = \phi(z^h) 
.. math::
   P^h = \psi(z^h) 

where, :math:`W^h \in \mathbb{R}^{|C_{H_h}| \times (|x| + |A^{h - 1}|)}`. The last layer (where :math:`h = |H|`) does not produce `A^{|H|}` as there is no further layer after it. Layer normalisation and dropout are performed in said order but not shown here for clarity. For indexing classes, each hierarchical level's label indexing starts from zero, and the hierarchical label is then represented as an ordered list of indices. The complete classification for each example is the concatenation of one-hot vectors from each level.

DB-BHCN+AWX
~~~~~~~~~~~~~~~~~~~~~~~~~
One large weakness of DB-BHCN is that it only strives to be as close to be hierarchically-compliant as possible, but cannot guarantee to be always so. As its name implies, this variant is a modified instance of our model that is then coupled with our implementation of the Adjacency Wrapping Matrix (AWX) layer from :cite:`masera2018awx`. In this variant, the local outputs branching from the flow are relegated to loss function minimisation duty. 

DB-BHCN requires small changes to be able to integrate with AWX, the most significant of which being the omission of the hierarchical loss function. As AWX already uses :math:`R`, it needs not learn hierarchical constraints through backward propagation. In addition, we have empirically found that the hierarchical loss conflicts with the AWX layer. In its place, we compute a binary cross-entropy (BCE) loss value from the AWX output and use it in conjunction with the existing local loss function. The local loss function on the other hand was found to contribute significantly to this variant's capability to quickly pick up performance early in training. Another adaptation is the usage of the Sigmoid activation function before feeding the last hidden layer to AWX. LogSoftmax activation is still used for the concatenated local outputs for use with the existing NLL-based local loss function.
