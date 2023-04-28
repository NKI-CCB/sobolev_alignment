# Sobolev Alignment

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/NKI-CCB/sobolev_alignment/main.svg)](https://results.pre-commit.ci/latest/github/NKI-CCB/sobolev_alignment/main)
[![codecov](https://codecov.io/gh/NKI-CCB/sobolev_alignment/branch/main/graph/badge.svg?token=GRLU3XBPO5)](https://codecov.io/gh/NKI-CCB/sobolev_alignment)

[badge-tests]: https://img.shields.io/github/actions/workflow/status/saroudant/sobolev_alignment/test.yaml?branch=main
[link-tests]: https://github.com/saroudant/sobolev_alignment/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/sobolev_alignment

This GitHub repository contains the implementation of Sobolev Alignment, a computational framework designed to align pre-clinical and tumor scRNA-seq data. Sobolev Alignment combines a deep generative model with a kernel method to detect non-linear processes that are shared by a source (e.g., cell line) and a target (e.g., tumor) dataset.

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.8 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

For the moment, no PyPI implementation is available (coming soon). The installation can be done in two steps.

### 1. Install Sobolev Alignment

You can install Sobolev Alignment and (almost) all dependencies using the following command:

```bash
pip install git+https://github.com/saroudant/sobolev_alignment.git@main
```

The resulting package is ready to use, but will use scikit-learn instead of Falkon, resulting in largely sub-optimal performances.

### 2. Install Falkon

To employ large-scale GPU-accelerated kernel methods, we turn to Falkon. The installation notice for Falkon is available on the [FalkonML documentation](https://falkonml.github.io/falkon/install.html). The previous installation procedure has already taken care of the various dependencies required for Falkon (i.e., cython, scipy and torch.)

## Frequent issues

### Issues with the compiler.

Due to incompatibilities between g++, gcc and cuda, the installation of FalkonML sometimes fails. The following elements can help alleviate potential issues:

-   Prior to installing Falkon, re-install torch 1.11.
-   Check compatibility between your cuda version and the one installed with torch.
-   Using cxx-compiler=1.2.0 (available on conda-forge) is compatible with cuda 11.3.

### Issues with Jaxlib (MacOS)

For Mac users, the jaxlib version installed from PyPI sometimes returns issues. We then advise to re-install jaxlib from condo, and subsequently re-install dcvi-tools:

```bash
mamba install jaxlib
mamba install scvi-tools
```

### Incompatibilities with numba

Errors are sometimes raised due to numba inconsistencies. The errors raised were due to clashes between different packages. Re-installing numba seem to have fixed the issues:

```bash
pip install numba --force-reinstall
```

Please feel free to contact the development team by e-mail or by creating an issue.

<!--
1) Install the latest release of `sobolev_alignment` from `PyPI <https://pypi.org/project/sobolev_alignment/>`_:

```bash
pip install sobolev_alignment
```
-->

## Workflow presentation

![Sobolev Alignment workflow](https://github.com/NKI-CCB/sobolev_alignment/blob/main/workflow.png)

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out at the following e-mail address: s [dot] mourragui [at] hubrecht [dot] eu.

## Citation

> [Identifying commonalities between cell lines and tumors at the single cell level using Sobolev Alignment of deep generative models, Mourragui et al, 2022, Biorxiv](https://www.biorxiv.org/content/10.1101/2022.03.08.483431v1)

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/saroudant/sobolev_alignment/issues
[changelog]: https://sobolev_alignment.readthedocs.io/latest/changelog.html
[link-docs]: https://sobolev-alignment.readthedocs.io/en/latest/
[link-api]: https://sobolev_alignment.readthedocs.io/en/latest/api.html
