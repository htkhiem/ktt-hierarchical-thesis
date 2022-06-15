.. Quickstart page.

Quickstart
==========

This guide will help you get started with training and exporting your first model based on an example dataset to an online inference system. Your resulting model can be used by a hypothetical e-commerce platform to give on-the-fly hierarchical product categorisation hints as a poster types their product name. It assumes that you have successfully downloaded the latest version of the system and installed all dependencies without error. If you have not done so, please refer to the previous page for instructions.

Data preparation
----------------

There are currently two ways to get data into the system:

    * Connecting an SQL database
    * Reading CSV, JSON, Parquet, Arrow or Feather files

The SQL approach is more advanced, so for this quickstart guide, we will instead use a flatfile. Bundled in the repository is an example dataset derived from `this CC0 dataset on Kaggle <https://www.kaggle.com/datasets/promptcloud/walmart-product-details-2020>`_, stored as ``datasets/Walmart_30k.parquet``. Although it had been cleaned up to a degree, it is still in its original schema and thus serves as a close example to flatfiles you would encounter in the wild. A peek at the file (using any Parquet-compatible tool, such as Python's ``pandas``) shows how the data inside are organised:

.. code-block:: bash

    idx                                                title                                        description  List Price  Sale Price            Brand                                           category
    0         La Costena Chipotle Peppers, 7 OZ (Pack of 12)  La Costena Chipotle Peppers, 7 OZ (Pack of 12)...       31.93       31.93  La Costeï¿½ï¿½a  [Food, Meal Solutions, Grains & Pasta, Canned ...
    1      Equate Triamcinolone Acetonide Nasal Allergy S...  Compare to Nasacort Allergy 24HR active ingred...       10.48       10.48           Equate  [Health, Equate, Equate Allergy, Equate Sinus ...
    2      AduroSmart ERIA Soft White Smart A19 Light Bul...  The Soft White ERIA A19 bulb (2700K) can be co...       10.99       10.99  AduroSmart ERIA  [Electronics, Smart Home, Smart Energy and Lig...
    3      24" Classic Adjustable Balloon Fender Set Chro...  Lowrider Fender Set 24" Classic Adjustable Chr...       38.59       38.59         lowrider  [Sports & Outdoors, Bikes, Bike Accessories, B...
    4      Elephant Shape Silicone Drinkware Portable Sil...   This is a kind of fine quality silicone cup l...        5.81        5.81           Anself  [Baby, Feeding, Sippy Cups: Alternatives to Pl...
    ...                                                  ...                                                ...         ...         ...              ...                                                ...
    29201                                      McCain Smiles  Add a wholesome side to your dinnertime meal w...        0.00        0.00           McCain            [Food, Frozen Foods, Frozen Vegetables]
    29202  Shock Sox Fork Seal Guards 29-36mm Fork Tube 4...  Shock Sox are a wraparound, neoprene sleeve th...       33.25       33.25        Shock Sox  [Sports & Outdoors, Bikes, Bike Components, Bi...
    29203                          Princes Gooseberries 300g    Gooseberries in syrup Princes Gooseberries 300g        8.88        8.88          Princes  [Food, Meal Solutions, Grains & Pasta, Canned ...
    29204  Create Ion Grace 3/4 Inches Straight Hair Iron...  Create ion grace straight is a 3/4 inches wide...       50.00       24.50       Create Ion  [Beauty, Hair Care, Hair Styling Tools, Flat I...
    29205  Green Bell Takuminowaza Two Way Ear Pick Brass...  Green bell ear wax remover 2-way screw & spoon...        6.00        4.20     Takuminowaza  [Beauty, Here for Every Beauty, Featured Shops...

    [29206 rows x 6 columns]

Since our model should be able to suggest which category to put a product into as the user types the product name, we will use the ``title`` column as our input to the model, and ``category`` as our training labels. Note how the ``category`` column is a list of strings, ordered by hierarchical depth (from the coarsest to the most detailed categorisation level). This is the common style that KTT uses. JSON files will use their own list syntax, while CSV files can accept different delimiters to split a single string into separate categories. Formats that natively support list datatypes for cells, such as Apache Arrow, Feather and Parquet, are directly supported and are most preferrable.

.. note::
   In addition, they are much lighter in size and significantly faster to parse as well as having additional capabilities such as out-of-core reading, so we highly recommend using them for your data storage outside of our system to lower storage costs (especially on services such as Amazon S3) and ease the data preparation process.

To input this flatfile into the system, we invoke the *flatfile adapter*:

.. code-block:: bash

   cd adapters
   python adapter_flat.py -v -t title -c category -d 2 --dvc ../datasets/Walmart_30k.parquet Walmart

An explanation of what these options and arguments do:

    * ``-v`` is the verbose argument. With this, the adapter prints more information about its process, which helps with understanding what it does.
    * ``-t`` specifies the column to use as the model's input. As we have previously discussed, we are using the ``title`` column. If left unspecified, ``title`` is also the default column name KTT uses. Of course, if you don't specify it and the flatfile does not have a ``title`` column, or it has one but the column is something else entirely and not what we need, expect some errors or poor performance.
    * ``-c`` likewise specifies the column name to use as classes (labels). It defaults to ``classes`` if left unspecified. Here, we want to use the ``category`` column, so of course we must specify it.
    * ``-d`` is the hierarchical depth to search to. By default, the adapter runs for two levels. This must not exceed the *minimum* depth of the dataset, i.e. the length of the shortest list in the label column.
    * ``--dvc`` tells the adapter to also add the resulting intermediate dataset to the Data Version Control system. With this, you can later add a remote of your choosing, push the dataset there and safely delete the local copy to save disk space. Training scripts will automatically pull them from said remote when needed.
    * The first argument is the path to the flatfile. `Walmart_30k.parquet` is bundled in `datasets`.
    * The second argument is the name of the resulting intermediate dataset.

After running the above commands, a new folder (``datasets/Walmart``) will be created, containing three Parquet files and a JSON file named ``hierarchy.json``.

.. note::

    As a quickstart guide, we won't delve too deeply into the specifics, but here are the main tasks an adapter performs:

    1. Read the flatfile or SQL data into memory, then rename columns and reformat the `classes` column if necessary.
    2. Build a list of pairwise-different labels on each level.
    3. Reconstruct the hierarchy from relationships found in the flatfile or corresponding SQL table.
    4. Generate metadata for specific models.
    5. CV-split the dataset into a training set, a validation set and a test set. This is done using random sampling.
    6. Write the three sets into a new folder within ``datasets`` as Parquet files (irrespective of input format) and the hierarchical metadata as a JSON file.
    
Training a model
----------------

Having generated a compatible model and the necessary metadata, we can now train a model on it. To keep everything lightweight for a quickstart guide, we will use a small CPU-based model called Tfidf + Hierarchical SGD (internal identifier ``tfidf_hsgd``). This model is capable of fully hierarchical classification and trains very quickly, albeit with limited accuracy for small datasets.

From ``./``, run the following command to train it:

.. code-block:: bash

    python train.py -m tfidf_hsgd Walmart
    
An explanation, again:

    * ``-m`` specifies which model to train. Here we use the internal identifier of the above model.
    * The one and only argument specifies the (internal) dataset name to train on. We have previously named our dataset ``Walmart`` (to differentiate it from the flatfile, which was named ``Walmart_30k``, so we'll use that name here.

.. note::
    
    Both the dataset-name argument and the ``-m`` option accept comma-separated lists of models and datasets. In other words, you can tell the training script to train on multiple datasets at once, train multiple models at once, or train multiple models, each on multiple datasets!
    
Once the command finishes, a ``tfidf_hsgd`` instance will have been trained on ``Walmart`` and saved in the ``weights/tfidf_hsgd/Walmart`` directory.

Exporting the trained model
---------------------------

With a trained model under our belt, we can export it into some sort of API server. Our goal is to have a minimal inference server set up by the end of this guide, so we will take advantage of the preset BentoML-powered inference system in KTT. Every bundled model in KTT has been prepared for usage by BentoML through special build scripts and service files.

Run the following command from from ``.\`` to export the newly trained model:

.. code-block:: bash

    python export.py -m 'tfidf_hsgd' -b Walmart
    
This command will then look for the latest saved instance of ``tfidf_hsgd`` trained on the ``Walmart`` dataset - note how the ``-m`` option once again present in this command and the argument is the dataset name. The additional ``-b`` argument tells the script to generate a BentoService and copy the necessary supporting data along. The resulting service will be located in the ``build`` folder.

.. warning::

    Unlike the other six models, ``tfidf_hsgd`` does not support exporting to other formats than BentoML. As such, the ``-b`` flag MUST always be specified when exporting this model.

Serving up a Bento
------------------

It's Bento(ML) time! KTT's BentoML inference pipelines relies on packaged systems known as BentoServices. These are the model graph themselves, plus the code needed to feed data in and extract data out of the model (i.e. a REST-to-model-to-REST routine), a lot of supporting files for peripheral subsystems such as a live performance monitoring dashboard (optional), and a version-frozen list of all dependencies.

As above, each and every bundled model in the system has their own Bento pipeline implemented. As such, in order to serve our trained model as a REST API, run the following command:

.. code-block:: bash

    bentoml serve build/tfidf_hsgd_Walmart/inference

The model will be served as a REST API at ``localhost:5000`` with the ``/predict`` endpoint. You can use the supplied ``test.py`` test script in ``onnx/bentoml`` to send a single request to it and check the results.

.. code-block:: bash
    
    cd onnx/bentoml
    python test.py

Or, if you prefer to send requests to it yourself, simply POST in the following format:

    - Content-Type: ``application/json``
    - Data: a JSON string with a single field ``text`` containing the textual input to be classified. For example: ``"{ \"text\": \"Classify this string\" }"``.

Shipping Bentos in a container
------------------------------

Having a BentoService online should be enough if you only plan on running it directly on the computer that trained it, or another that you are sure has had all dependencies correctly installed. However, this is hard to maintain, especially as libraries may change over time, causing breakages. Furthermore, downloading all dependencies over and over again is not a small task, and certain libraries may even become no longer available on the cloud for you to download (in extreme cases that is). Furthermore, KTT provides additional monitoring capabilities for its live inference systems that would be much, much more easily run as a single ``docker-compose`` network instead of separate manually-started processes (although the monitoring part isn't within the scopes of this Quickstart guide).

As such, it is preferrable that we find a way to keep an offline backup of such dependencies, frozen to the exact version used to train and export the model. This is where Dockerisation comes in.

First, ensure that you have Docker correctly installed (and its daemon running) on your local system and that your usee account has the necessary privileges. Then, Dockerising a BentoService is a one-command affair:

.. code-block:: bash
    
    docker image build ./build/tfidf_hsgd_Walmart_005
    
This might take a while as Docker builds a Debian-based system with our model. Once it's done, check the list of images on your system and note down the Image ID of the newly-built Bento container.

.. code-block:: bash

    ❯ docker images
    REPOSITORY             TAG                                IMAGE ID       CREATED       SIZE
    tfidf_hsgd_Walmart_005 ................                   fbbf4a810b58   ...           ...
    
Here the Image ID is ``fbbf4a810b58``. We can now fire the image up:

.. code-block:: bash

    docker run -p 5000:5000 fbbf4a810b58
    
The ``-p 5000:5000`` argument forwards the host's port 5000 to the container's corresponding port. This allows requests from the outside to be directed to the container, and the container's response to be in turn directed back to the outside. You can test this using the same test script we have mentioned above.

.. note::

   The Dockerisation process is a bit different for BentoServices built with monitoring capabilities enabled. Since they use multiple Docker containers, KTT will also generate a ``docker-compose.yaml`` file for them. Simply navigate to the built service's folder (where the ``docker-compose.yaml`` file is) and hit ``docker-compose up``.

If all goes well, congratulations! You now have a fully self-contained Docker image of your newly-trained hierarchical classification model.
