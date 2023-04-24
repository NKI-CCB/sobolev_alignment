"""
Kernel Ridge Regression (KRR) model search.

@author: Soufiane Mourragui

Pipeline to perform model selection for the Kernel Ridge Regression (KRR) models,
employing the protocol presented in the paper, i.e.,:
- Selecting sigma as the value yielding an average of 0.5 for the Gaussian kernel.
- Selecting model with lowest training error on input data (trained on artificial
data).
"""

import gc
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from sklearn.model_selection import ParameterGrid

# from .sobolev_alignment import SobolevAlignment

DEFAULT_KRR_PARAMS = {
    "method": ["falkon"],
    "kernel": ["matern"],
    "M": [250],
    "penalization": np.logspace(-8, 2, 11),
    "kernel_params": [
        {"sigma": s, "nu": n}
        for s in [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            12.5,
            15.0,
            17.5,
            20.0,
            22.5,
            25.0,
            30.0,
            40.0,
            50.0,
        ]
        for n in [0.5, 1.5, 2.5, np.inf]
    ],
    "maxiter": [20],
    "falkon_options": [
        {
            # 'max_cpu_mem': 2**(8*4.8),
            "never_store_kernel": False,
            # 'max_gpu_mem': 2**(8*4.2),
            "min_cuda_iter_size_32": np.iinfo(np.int64).max,
            "min_cuda_iter_size_64": np.iinfo(np.int64).max,
            "num_fmm_streams": 6,
        }
    ],
}


def model_selection_nu(
    X_source: AnnData,
    X_target: AnnData,
    sobolev_alignment_clf,
    sigma: float,
    M: int = 250,
    test_error_size: int = -1,
):
    r"""
    Select the optimal $nu$ parameter.

    Select the optimal $nu$ parameter (Matérn kernel) by measuring the Spearman correlation
    for different values of $nu$ and penalization, and selecting the $nu$ with the
    highest correlation.

    Parameters
    ----------
    X_source: AnnData
        Source dataset.

    X_target: AnnData
        Target dataset.

    sobolev_alignment_clf: SobolevAlignment
        SobolevAlignment instance with scVI models trained. Used to find optimal
        $nu$ parameter on the KRR regression step.

    sigma: float
        $\sigma$ parameter in KRR.

    M: int, default to 250
        Number of anchor points to use in the KRR approximation. A larger M
        typically improves the prediction, but at the cost of longer compute
        time and memory cost.

    test_error_size: float, default to -1
        Number of input points to be considered when computing the error. Input (X_source
        and X_target) are not used to train the KRR (artificial points are) and are
        acting as proxy for validation set. Setting test_error_size=-1 would lead
        to using the complete input data

    Returns
    -------
        DataFrame with spearman correlation on source and target data for various
        hyper-parameter values.
    """
    krr_param_grid = _make_hyperparameters_grid(sigma, M)

    latent_results_krr_error_df = {}

    # Generate artificial samples
    sobolev_alignment_clf.fit(
        X_source=X_source, X_target=X_target, fit_vae=False, sample_artificial=True, krr_approx=False
    )

    for krr_params in krr_param_grid:
        param_id = "kernel_{}_M_{}_penalization_{}_params_{}".format(
            krr_params["kernel"],
            krr_params["M"],
            krr_params["penalization"],
            "$".join([f"{e}:{f}" for e, f in krr_params["kernel_params"].items()]),
        )

        sobolev_alignment_clf.krr_params = {"source": deepcopy(krr_params), "target": deepcopy(krr_params)}

        print("\t START %s" % (param_id), flush=True)
        sobolev_alignment_clf.fit(
            X_source=X_source, X_target=X_target, fit_vae=False, sample_artificial=False, krr_approx=True
        )
        gc.collect()

        # Compute_error
        krr_approx_error = sobolev_alignment_clf.compute_error(size=test_error_size)
        processed_error_df = {x: _process_error_df(df) for x, df in krr_approx_error.items()}
        processed_latent_error_df = {x: df[0] for x, df in processed_error_df.items()}
        processed_latent_error_df = pd.concat(processed_latent_error_df)
        latent_results_krr_error_df[param_id] = processed_latent_error_df
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Process latent_factor_error
    latent_error_df = pd.concat(latent_results_krr_error_df)
    latent_error_df.index.names = ["param_id", "data_source", "data_generation"]
    latent_error_df = latent_error_df.reset_index()
    latent_error_df = latent_error_df.reset_index().pivot_table(
        values="spearmanr", index=["data_source", "param_id"], columns=["data_generation"]
    )
    latent_spearman_df = pd.concat({x: latent_error_df.loc[x]["input"] for x in ["source", "target"]}, axis=1)
    latent_spearman_df["combined"] = np.sum(latent_spearman_df, axis=1) / latent_spearman_df.shape[1]
    latent_spearman_df = latent_spearman_df.sort_values("combined", ascending=False)

    return latent_spearman_df


def model_alignment_penalization(
    X_data: AnnData, data_source: str, sobolev_alignment_clf, sigma: float, optimal_nu: float, M: int = 250
):
    r"""
    $\\sigma$ and $\nu$ selection.

    Select the optimal penalization parameter given $\\sigma$ and $\nu$ by aligning the
    data_source model to itself and measuring the principal angles.
    Intuitively, aligning the model to itself must yield high principal angles. Low
    values indicate over-fitting of the KRR.

    Parameters
    ----------
    X_data: AnnData
        Dataset to employ.

    data_source: str, 'source' or 'target'
        Name of the data stream in SobolevAlignment parameters.

    sobolev_alignment_clf: SobolevAlignment
        SobolevAlignment instance with scVI models trained. Used to find optimal
        $nu$ parameter on the KRR regression step.

    sigma: float
        $\\sigma$ parameter in KRR.

    optimal_nu: float
        Value of $\nu$ (Falkon) to be used in the optimization. Can be established
        using model_selection_nu

    M: int, default to 250
        Number of anchor points to use in the KRR approximation. A larger M
        typically improves the prediction, but at the cost of longer compute
        time and memory cost.

    Returns
    -------
        DataFrame with principal angles between the same models.
    """
    _clf = deepcopy(sobolev_alignment_clf)
    supp_data_source = "target" if data_source == "source" else "source"
    _clf.scvi_models[supp_data_source] = sobolev_alignment_clf.scvi_models[data_source]
    _clf.scvi_batch_keys_[supp_data_source] = sobolev_alignment_clf.scvi_batch_keys_[data_source]
    gc.collect()

    # Artificial sampling
    _clf.n_jobs = 1
    _clf.batch_name[supp_data_source] = _clf.batch_name[data_source]
    _clf.continuous_covariate_names[supp_data_source] = _clf.continuous_covariate_names[data_source]
    _clf.fit(X_source=X_data, X_target=X_data, fit_vae=False, sample_artificial=True, krr_approx=False)

    krr_param_grid = _make_hyperparameters_grid(sigma, M, [optimal_nu])
    principal_angles_df = {}
    for krr_params in krr_param_grid:
        param_id = "kernel_{}_M_{}_penalization_{}_params_{}".format(
            krr_params["kernel"],
            krr_params["M"],
            krr_params["penalization"],
            "$".join([f"{e}:{f}" for e, f in krr_params["kernel_params"].items()]),
        )

        _clf.krr_params = {"source": krr_params, "target": krr_params}

        print("\t START %s" % (param_id), flush=True)
        _clf.fit(X_source=X_data, X_target=X_data, fit_vae=False, sample_artificial=False, krr_approx=True)
        gc.collect()

        # Compute_error
        principal_angles_df[param_id] = _clf.principal_angles
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return principal_angles_df


def _make_hyperparameters_grid(sigma, M, nu_values=None):
    nu_values = nu_values if nu_values else [0.5, 1.5, 2.5, np.inf]
    krr_param_possibilities = deepcopy(DEFAULT_KRR_PARAMS)
    krr_param_possibilities["kernel_params"] = [{"sigma": sigma, "nu": n} for n in nu_values]
    krr_param_possibilities["M"] = [M]
    return ParameterGrid(krr_param_possibilities)


def _process_error_df(df):
    latent_error_df = pd.DataFrame(df["latent"])
    factor_error_df = pd.concat({x: pd.DataFrame(df["factor"][x]) for x in df["factor"]})
    return [latent_error_df, factor_error_df]
