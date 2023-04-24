"""
Generate artificial samples.

@author: Soufiane Mourragui

Generate samples using scVI decoder from a multivariate gaussian noise.
This module generates the training data used to approximate the VAE encoding
functions by Matérn kernel machines.

"""

import numpy as np
import pandas as pd
import scvi
import torch
from joblib import Parallel, delayed


def generate_samples(
    sample_size: int,
    batch_names: list,
    covariates_values: list,
    lib_size: dict,
    model: scvi.model.SCVI,
    batch_key_dict: dict,
    return_dist: bool = False,
):
    """
    Generate artificial gene expression profiles.

    Note to developers: this method has been designed to be used with scvi-tools classes. Other VAE
    implementations may break here.

    Parameters
    ----------
    sample_size: int
        Number of samples to generate.

    batch_names: list or np.ndarray, default to None
        List or array with sample_size str values indicating the batch of each sample.

    covariate_values: list or np.ndarray, default to None
        List or array with sample_size float values indicating the covariate values of each
        sample to generate (as for training scVI model).

    lib_size
        Dictionary of mean library size per batch.

    model
        scVI model which decoder is here exploited to generate samples.

    batch_key_dict
        Dictionary linking the values of the batch (scVI) and the key used in scVI.

    return_dist: bool, default to False
        Whether to return the distribution parameters (True) or samples from this distribution (False).

    Returns
    -------
    If return_dist if False, torch.Tensor (on CPU) with artificial samples in the rows.
    If return_dist if True, torch.Tensor with distribution parameters (following scVI
    order) and one torch.Tensor with artificial samples in the rows (CPU).
    """
    # Retrieve batch name from scVI one-hot-encoding
    if batch_names is not None:
        batch_name_ids = [batch_key_dict[n] for n in batch_names]
        batch_name_ids = torch.Tensor(np.array(batch_name_ids).reshape(-1, 1))
        # Recover log library size (exponential)
        lib_size_samples = np.array([np.random.choice(lib_size[n], 1)[0] for n in batch_names])
        lib_size_samples = np.log(lib_size_samples)
    else:
        batch_name_ids = None
        lib_size_samples = np.random.choice(np.array(lib_size).flatten(), sample_size)
        lib_size_samples = np.log(lib_size_samples)

    # Process covariates
    if covariates_values is None:
        cont_covs = None
    elif type(covariates_values) is pd.DataFrame:
        cont_covs = torch.Tensor(covariates_values.values.astype(float))
    elif type(covariates_values) is np.array:
        cont_covs = torch.Tensor(covariates_values)

    # Generate random noise
    z = torch.Tensor(np.random.normal(size=(int(sample_size), model.init_params_["non_kwargs"]["n_latent"])))
    dist_param_samples = model.module.generative(
        z=z,
        library=torch.Tensor(np.array(lib_size_samples).reshape(-1, 1)),
        batch_index=batch_name_ids,
        cont_covs=cont_covs,
    )

    # Sample from distribution
    samples = dist_param_samples["px"].sample()

    if return_dist:
        return dist_param_samples, samples
    else:
        return samples.cpu()


def parallel_generate_samples(
    sample_size,
    batch_names,
    covariates_values,
    lib_size,
    model,
    batch_key_dict: dict = None,
    return_dist: bool = False,
    batch_size=10**3,
    n_jobs=1,
):
    """
    Generate artificial gene expression profiles.

    Wrapper of parallelize generate_samples, running several threads in parallel.
    <b>Note to developers</b>: this function needs to be changed if applied to other VAE model
    than scVI.

    Parameters
    ----------
    sample_size: int
        Number of samples to generate.

    batch_names: list or np.ndarray, default to None
        List or array with sample_size str values indicating the batch of each sample.

    covariate_values: list or np.ndarray, default to None
        List or array with sample_size float values indicating the covariate values of each
        sample to generate (as for training scVI model).

    lib_size
        Dictionary of mean library size per batch.

    model
        scVI model which decoder is here exploited to generate samples.

    batch_key_dict
        Dictionary linking the values of the batch (scVI) and the key used in scVI.

    return_dist: bool, default to False
        Whether to return the distribution parameters (True) or samples from this distribution (False).

    batch_size: int, default to 10**3
        Number of sample to generate per batch.

    n_jobs: int, default to 1
        Number of threads to launch. n_jobs=-1 will launch as many threads as there are CPUs available.

    Returns
    -------
    If return_dist if False, torch.Tensor (on CPU) with artificial samples in the rows.
    If return_dist if True, torch.Tensor with distribution parameters (following scVI
    order) and one torch.Tensor with artificial samples in the rows (CPU).
    """
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(generate_samples)(
            sample_size=batch_size,
            batch_names=batch_names[i : i + batch_size] if batch_names is not None else None,
            covariates_values=covariates_values[i : i + batch_size] if covariates_values is not None else None,
            lib_size=lib_size,
            model=model,
            batch_key_dict=batch_key_dict,
            return_dist=return_dist,
        )
        for i in range(0, sample_size, batch_size)
    )

    if return_dist:
        return results

    return torch.cat(results)
