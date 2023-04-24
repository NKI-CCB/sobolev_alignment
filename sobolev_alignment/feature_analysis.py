"""
Feature analysis.

@author: Soufiane Mourragui

This modules contains all the codes used in the Taylor expansion for the Gaussian/Matern
kernel.
"""

import gc
import logging
from functools import reduce
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed


def higher_order_contribution(
    d: int,
    data: np.array,
    sample_offset: np.array,
    gene_names: list,
    gamma: float,
    n_jobs: int = 1,
    return_matrix: bool = False,
):
    r"""Compute the features corresponding to the Taylor expansion of the kernel.

    Compute the features corresponding to the Taylor expansion of the kernel, i.e. $x_j exp^{-\gamma xx^T}$ for
    linear features. Returns a sparse pandas DataFrame containing all the features (columns) by samples (rows).
    We here critically rely on the sparsity of the data-matrix to speed up computations. The current implementation
    is relevant in two cases:
    -When dimensionality is small
    -When data is sparse.

    High-dimensional and dense data matrices would lead to a significant over-head without computational gains,
    and could benefit from another implementation strategy.

    Parameters
    ----------
    d: int
        Order of the features to compute, e.g. 1 for linear, 2 for interaction terms.

    data: np.array
        Data to compute features on, samples in the rows and genes (features) in the columns.

    sample_offset: np.array
        Offset of each sample from data.

    gene_names: list
        Names of each columns in data ; corresponds to features naming.

    gamma: float
        Value of the gamma parameter for Matérn kernel.

    n_jobs: int, default to 1
        Number of concurrent threads to use. -1 will use all CPU cores possible.
        WARNING: for d >= 2 and a large number of genes, the routine can be memory-intensive and a high n_jobs
        could lead to crash.

    return_matrix: bool, default to False
        If True, then returns simply the feature-matrix without feature-naming. In cases when feature names
        are not relevant (e.g. computing the proportion of non-linearities), return_matrix=True can help
        speed-up the process.

    Returns
    -------
    pd.DataFrame
        Sparse dataframe with samples in the rows and named features in the columns.
        For instance, when d=1, returns each column of data scaled by RKHS normalisation factor and multiplied
        by offset value.
    """
    # Exploits sparsity of scRNA-seq data (even more handy when d >= 2)
    # Note to future user: this can be an issue if data is not sparse
    sparse_data = scipy.sparse.csc_matrix(data)

    # Compute features by iterating over possible combinations
    logging.info("\t START FEATURES")
    combinations_features = Parallel(n_jobs=n_jobs, verbose=1, max_nbytes=1e6, pre_dispatch=int(1.5 * n_jobs))(
        delayed(combinatorial_product)(sparse_data, x, gamma)
        for x in combinations_with_replacement(np.arange(sparse_data.shape[1]), r=d)
    )
    gc.collect()

    # Combine features and multiply columns by offset.
    logging.info("\t START CONCATENATION")
    logging.info("\t\t START STACKING")
    combinations_features = scipy.sparse.hstack(combinations_features, format="csc")
    logging.info("\t\t START PRODUCT")
    combinations_features = scipy.sparse.diags(sample_offset).dot(combinations_features)
    gc.collect()
    if return_matrix:
        return combinations_features

    # Return names of each features.
    logging.info("\t\t FIND NAMES")
    combinations_names = Parallel(
        n_jobs=min(5, n_jobs), verbose=1, max_nbytes=1e4, pre_dispatch=int(1.5 * min(5, n_jobs))
    )(delayed(_interaction_name)(x) for x in combinations_with_replacement(gene_names, r=d))

    return pd.DataFrame.sparse.from_spmatrix(data=combinations_features, columns=combinations_names)


def _combination_to_idx(idx, p):
    r"""Transform a combination (tuple of feature idx) into an indicative function.

    Parameters
    ----------
    idx: tuple
        Combination of features in the form of a tuple. <br/>
        E.g. for 6 genes, (5,1) corresponds to the product of 1 and 5 and returns
        (0,1,0,0,0,1), while (1,2,3,2) will yield (0,1,2,1,0,0). <br/>
        <b>WARNING:</b> start at 0.

    p: int
        Number of genes (features) in the dataset.

    Returns
    -------
    np.array
        Indicative function of the combination
    """
    return np.array([np.sum(np.array(idx) == i) for i in range(p)])


def basis(x, k, gamma):
    r"""Compute the basis function for a single gene, except offset term.

    Parameters
    ----------
    x: np.array
        Column vector (each row corresponds to a sample).

    k: int
        Order to compute.

    gamma: float
        Parameter of Matérn kernel.

    Returns
    -------
    np.array
        Value of the higher order feature.
    """
    if k == 0:
        return np.ones(x.shape[0])

    product = x
    for _ in range(1, k):
        product = x.multiply(product)
    coef = np.power(2 * gamma, k / 2) / np.sqrt(scipy.math.factorial(k))

    return coef * product


def combinatorial_product(x, idx, gamma):
    """
    Compute the basis function for a single gene, except offset term.

    Parameters
    ----------
    x: np.array
        Data matrix with samples in the rows and genes in the columns

    idx: tuple
        Combinations, i.e. tuple of features to take into account.

    gamma: float
        Parameter of Matérn kernel.

    Returns
    -------
    scipy.sparse.csc_matrix
        Values of the higher order feature.
    """
    # Iterate over all genes and compute the feature weight by multiplication
    prod = [basis(x[:, i], k, gamma) for i, k in enumerate(_combination_to_idx(idx, x.shape[1])) if k > 0]
    if len(prod) == 0:
        return 1

    return reduce(scipy.sparse.csc_matrix.multiply, prod)


def _interaction_name(gene_combi):
    combin_name = [f"{g}^{r}" for g, r in zip(*np.unique(gene_combi, return_counts=True))]
    return "*".join(combin_name) if len(combin_name) > 0 else "1"


def _higher_order_interaction_wrapper(data, x, gamma, gene_names):
    return [combinatorial_product(data, x, gamma), _interaction_name(gene_names, _combination_to_idx(x, data.shape[1]))]


def _compute_offset(data, gamma):
    r"""Compute the sample-level offset values, i.e. $\exp -\gamma xx^T$.

    Parameters
    ----------
    data: np.array
        Data to compute features on, samples in the rows and genes (features) in the columns.

    gamma: float
        Value of the gamma parameter for Matérn kernel.

    Returns
    -------
    np.array
        One-dimensional vector with offset values of all samples.
    """
    sample_offset = np.linalg.norm(data, axis=1)
    return np.exp(-gamma * np.power(sample_offset, 2))
