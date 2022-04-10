.. Command documentation page.

CLI usage
===================================================

Individual components in our system are controlled via command-line arguments. This page details all available options for each component.

Training component
------------------

.. click:: train:main
   :prog: python3 train.py

Export component
------------------

.. click:: export:main
   :prog: python3 train.py


Adapters
--------

Flat-file adapter
~~~~~~~~~~~~~~~~~

.. click:: adapters.adapter_flat:main
   :prog: python3 adapter_flat.py

SQL adapter
~~~~~~~~~~~~~~~~~

.. click:: adapters.adapter_sql:main
   :prog: python3 adapter_sql.py
