.. Command documentation page.

CLI usage
===================================================

Individual components in our system are controlled via command-line arguments. This page details all available options for each component.

.. click:: train:main
   :prog: python3 train.py

.. click:: export:main
   :prog: python3 train.py


Adapters
--------

.. _adapter-sql:

.. click:: adapters.adapter_flat:main
   :prog: python3 adapter_flat.py

.. _adapter-flat:

.. click:: adapters.adapter_sql:main
   :prog: python3 adapter_sql.py
