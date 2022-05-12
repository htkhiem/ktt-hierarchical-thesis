DistilBERT
==========

The original English-language BERT has two models:  ``bert_base`` featuring 12 encoders with 12 bidirectional self-attention heads, and the ``bert_large`` featuring 24 encoders with 16 bidirectional self-attention heads. While these models lead to significant improvements over the previous state-of-the-art, they often have several hundred million parameters. Operating these large models in on-the-edge and/or under constrained computational training or inference budgets proved to be a challenge. Using the technique known as *knowledge distillation*, researchers propose a method to pre-train a smaller general-purpose language representation model known as DistilBERT - a model that retains BERT's architecture while being both lighter and faster, which can then be fine-tuned with good performances on a wide range of tasks like its larger counterparts.

While keeping the general architecture of BERT, DistilBERT is optimised by reducing *the number of layers*, as it has bigger impacts in the Transformer model architecture than other factors such as the variation on the last dimension of the tensor :cite:`1910-01108`. In addition, the right initialisation for the sub-network to converge is found by taking advantage of the common dimensionality between the teacher and student networks. This results in a general-purpose pre-trained version of BERT that is 40% smaller, 60% faster and retains 97% of the language understanding capabilities :cite:`1910-01108`, thus it becomes a compelling option for edge applications.

API
---

.. automodule:: utils.encoders.distilbert
	:members:
	:special-members:

