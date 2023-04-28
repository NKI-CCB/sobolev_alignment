.. Sobolev Alignment documentation master file, created by
   sphinx-quickstart on Thu Jan 19 21:16:54 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|CI|

Sobolev Alignment: aligning deep general models with large-scale kernel machines
=============================================

.. image:: https://github.com/NKI-CCB/sobolev_alignment/blob/main/workflow.png
    :width: 600px
    :align: center

Sobolev Alignment identifies commonalities between cell lines and tumors at the single cell level using Sobolev Alignment of deep generative models. Using scVI as a backbone for generative models (VAE), latent variable models are approximated by means of large-scale kernel machines, using FalkonML. This allows a systematic appro

.. toctree::
   :maxdepth: 3
   :caption: Contents:

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   sobolev_alignment
   krr_approx
   krr_model_selection
   feature_analysis


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |CI| image:: https://img.shields.io/github/actions/workflow/status/saroudant/sobolev_alignment/test.yaml?branch=main
   :target: https://github.com/NKI-CCB/sobolev_alignment/actions
   :alt: CI
