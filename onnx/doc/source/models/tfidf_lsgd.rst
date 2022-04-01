.. Tfidf + Leaf SGD documentation

Tf-idf + Leaf SGD
=========================

API
---

.. autoclass:: models.tfidf_lsgd.Tfidf_LSGD
   :members:
   :special-members:

Theory
------

A standard combination between the TF-IDF vectorization algorithm and a machine learning classifier based around stochastic gradient descent. This is a generic classifier and through different loss functions can represent different machine learning algorithms. For example, using a hinge loss would turn it into a multiclass-capable linear SVM :cite:`evgeniou1999support`. 
An example computation graph for the classifier is given below.

.. image:: tfidf-lsgd.svg
   :width: 320
   :alt: Topology of the Leaf SGD classifier.

In our experiment, we used a modified Huber loss, a classification-centric variant of the original Huber loss :cite:`zhang2004solving`. It is expressed as:

.. math::
    L(y, f(x)) = 
    \begin{cases}
    max(0, 1-yf(x))^2 &\text{for } yf(x) \geq -1\\
    -4yf(x) &\text{otherwise}
    \end{cases}


where :math:`y` is the target and :math:`f(x)` is the output of our model, here labelled as :math:`f`.
After preliminary testing, our implementation is now based around the **Scikit-learn** Python library :cite:`scikit-learn` and NLTK :cite:`bird2009natural`, a robust stemming library (specifically, we used their English SnowballStemmer with standard English stop-word removal). Input text first goes through stemming and stop-word removal, then through a TF-IDF vectoriser fitted against the general corpus with a cut-off frequency of 50, then to the classifier, which *classifies at the leaf level of the hierarchy}* In case multi-string examples are used (such as categorising products based on both their title and description), each string can go through its stemmer-vectoriser pipeline (possibly with differing hyperparameters); the outputs are then multiplied with weights that allow us to control the significance of each string, and lastly, concatenated. All major stages are packaged into **Scikit-learn** pipeline stages, which are then queued in a serialisable pipeline. This serialisable pipeline together with its trained parameters can be saved to disk for later inference.

To further aid in performance regarding class example count imbalance, we further weight the loss functions such that classes with fewer examples are given more priority against more frequently-encountered classes, which should prevent the classifier from resorting to population-based guessing. The specific weight for each class is computed using the following formula:

.. math::
    W_c = N / (C \times N_c)

where :math:`W_C` is the weight for examples of class :math:`c`, :math:`N` is the total example count for the entire training set, :math:`C` is the number of classes (in all levels) and :math:`N_c` is the number of examples of class :math:`c`.

