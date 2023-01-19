# sobolev_alignment

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/saroudant/sobolev_alignment/test.yaml?branch=main
[link-tests]: https://github.com/saroudant/sobolev_alignment/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/sobolev_alignment

Sobolev alignment of deep probabilistic models for comparing single cell profiles

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.8 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install sobolev_alignment:

<!--
1) Install the latest release of `sobolev_alignment` from `PyPI <https://pypi.org/project/sobolev_alignment/>`_:

```bash
pip install sobolev_alignment
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/saroudant/sobolev_alignment.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Added

master_doc = 'index' to docs/source/conf.py
sphinx-build -b html docs/ docs/html
sphinx-quickstart docs
cd docs
make html

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/saroudant/sobolev_alignment/issues
[changelog]: https://sobolev_alignment.readthedocs.io/latest/changelog.html
[link-docs]: https://sobolev_alignment.readthedocs.io
[link-api]: https://sobolev_alignment.readthedocs.io/latest/api.html
