.. Tfidf + Hierarchy SGD documentation

Tf-idf + Hierarchy SGD
=========================

API
---

.. autoclass:: models.Tfidf_HSGD
   :members:
   :special-members:

Configuration schema
--------------------

The configuration for this model defines the following hyperparameters:

* ``loss``: Type of loss function for use with the ``SGDClassifier``. See `the related documentation <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html>`_ for more details.
* ``max_iter``: The maximum number of iterations to train the ``SGDClassifier`` over.
* ``min_df``: The minimum number of occurences a token must have in the training dataset for it to be included in the ``tfidf`` vectoriser's vocabulary.

Default tuning configuration
----------------------------

.. code-block:: json

	"tfidf_hsgd": {
        "display_name": "Tfidf + HierarchicalSGD",
		"range": {
        	"loss": "modified_huber",
            "max_iter": [500, 1000, 2000],
            "min_df": [1, 2, 4, 8, 16, 32, 64, 128]
        },
        "mode": {
        	"loss": "fixed",
            "max_iter": "choice",
            "min_df": "choice"
        }
    }

Theory
------

This is an evolution of the model described in the `Tf-idf + Leaf SGD <tfidf_lsgd.html>`_ page. Instead of having one classifier predicting the leaf layer, we now construct a matching hierarchy of classifiers, one for each non-leaf node. For this, we enlist the help of **sklearn-hierarchical-classification** :cite:`globalit74online`, a library that works with **Scikit-learn** to construct a classifier hierarchy which itself is presented as a big classifier. It takes in a hierarchy, presented in a specific format, and a base classifier, which it would then use for each node in the hierarchy. Data is first split based on their leaf nodes (which dictates the path to the root, which in turn dictates which classifier(s) this example goes through). The split subsets are then fed to their corresponding node classifiers and trained separately. The resulting hierarchical classifier outputs leaf nodes, but the library can be modified at the source level to output the entire path for use with variable-depth termination pseudoclasses. However, since our example case prefers predictions down to the leaf nodes, this was not within our scope. Overall, this model belongs to the per-node approach.

