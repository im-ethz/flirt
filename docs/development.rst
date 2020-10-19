Development
===========

Setup
-----

Navigate to the downloaded FLIRT directory and install it via
``pip install -e .`` to install flirt to your local development
environment. You should use the editable option (``-e``) so you do not
have to reinstall it after every change.

Hint: to reload a module in an interactive session you can use

::

   import importlib
   importlib.reload(flirt.hrv)

Documentation
-------------

FLIRT uses `Sphinx <https://www.sphinx-doc.org>`__ for documentation.

Source Code should be documented using the `numpy docstring
format <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`__.