.. Tfidf + Hierarchy SGD documentation

Tf-idf + Hierarchy SGD
=========================

API
---

.. autoclass:: models.Tfidf_HSGD
   :members:
   :special-members:

Theory
------

This is an evolution of the model described in the `Tf-idf + Leaf SGD <tfidf_lsgd.html>`_ page. Instead of having one classifier predicting the leaf layer, we now construct a matching hierarchy of classifiers, one for each non-leaf node. For this, we enlist the help of **sklearn-hierarchical-classification** :cite:`globalit74online`, a library that works with **Scikit-learn** to construct a classifier hierarchy which itself is presented as a big classifier. It takes in a hierarchy, presented in a specific format, and a base classifier, which it would then use for each node in the hierarchy. Data is first split based on their leaf nodes (which dictates the path to the root, which in turn dictates which classifier(s) this example goes through). The split subsets are then fed to their corresponding node classifiers and trained separately. The resulting hierarchical classifier outputs leaf nodes, but the library can be modified at the source level to output the entire path for use with variable-depth termination pseudoclasses. However, since our example case prefers predictions down to the leaf nodes, this was not within our scope. Overall, this model belongs to the per-node approach.

