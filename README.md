# Sobolev Alignment

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

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

For the moment, no PyPI implementation is available. Falkon must be installed prior to installing Sobolev Alignment. The installation notice for Falkon is available on the [FalkonML documentation](https://falkonml.github.io/falkon/install.html).

Once Falkon has been installed, the following command will automatically sobolev_alignment, alongside the remaining dependencies (i.e., scvi-tools):

```bash
pip install git+https://github.com/saroudant/sobolev_alignment.git@main
```

We are currently working on a PyPI release.

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

> [Identifying commonalities between cell lines and tumors at the single cell level using Sobolev Alignment of deep generative models, Mourragui et al, 2022, Biorxiv][https://www.biorxiv.org/content/10.1101/2022.03.08.483431v1]

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/saroudant/sobolev_alignment/issues
[changelog]: https://sobolev_alignment.readthedocs.io/latest/changelog.html
[link-docs]: https://sobolev-alignment.readthedocs.io/en/latest/
[link-api]: https://sobolev_alignment.readthedocs.io/en/latest/api.html
