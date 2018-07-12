.. DiGeR-FD documentation master file, created by
   sphinx-quickstart on Mon Jun 11 16:55:03 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DiGeR-FD's documentation!
====================================

Documentation for the Code
**************************

.. automodule:: diger_fd


useful #1 -- auto members
=========================

This is something I want to say that is not in the docstring.

.. automodule:: diger_fd.file1
   :members:

useful #2 -- explicit members
=============================

This is something I want to say that is not in the docstring.

.. automodule:: diger_fd.file2
   :members: public_fn_with_sphinxy_docstring, _private_fn_with_docstring

.. autoclass:: MyPublicClass
   :members: get_foobar, _get_baz

Inheritance Diagram
===================

.. inheritance-diagram:: file2

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
