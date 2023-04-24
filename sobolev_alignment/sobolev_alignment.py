"""
Sobolev Alignment.

@author: Soufiane Mourragui

References
----------
Mourragui et al, Identifying commonalities between cell lines and tumors at the single cell level using
Sobolev Alignment of deep generative models, Biorxiv, 2022.
Lopez et al, Deep generative modeling for single-cell transcriptomics, Nature Methods, 2018.
Meanti et al, Kernel methods through the roof: handling billions of points efficiently, NeurIPS, 2020.
"""

import gc
import logging
import os
import re
from pickle import dump, load

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scvi
import seaborn as sns
import torch
from anndata import AnnData
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from ._scvi_default_params import SCVI_MODEL_PARAMS, SCVI_PLAN_PARAMS, SCVI_TRAIN_PARAMS
from .feature_analysis import _compute_offset, higher_order_contribution
from .generate_artificial_sample import parallel_generate_samples
from .interpolated_features import compute_optimal_tau, project_on_interpolate_PV
from .kernel_operations import mat_inv_sqrt
from .krr_approx import KRRApprox
from .krr_model_selection import model_alignment_penalization, model_selection_nu
from .multi_krr_approx import MultiKRRApprox
from .scvi_model_search import DEFAULT_HYPEROPT_SPACE, model_selection

# Default library size used when re-scaling artificial data
DEFAULT_LIB_SIZE = 10**3


class SobolevAlignment:
    """
    Sobolev Alignment implementation.

    Main class for Sobolev Alignment, which wraps all the different operations presented in Sobolev Alignment procedure:
    - Model selection (scVI and KRR)
    - scVI models training.
    - Synthetic models generations.
    - KRR approximation.
    - Alignment of KRR models.
    """

    default_scvi_params = {"model": {}, "plan": {}, "train": {}}

    def __init__(
        self,
        source_batch_name: str = None,
        target_batch_name: str = None,
        continuous_covariate_names: list = None,
        source_scvi_params: dict = None,
        target_scvi_params: dict = None,
        source_krr_params: dict = None,
        target_krr_params: dict = None,
        n_artificial_samples: int = 10**5,
        n_samples_per_sample_batch: int = 10**5,
        frac_save_artificial: float = 0.1,
        save_mmap: str = None,
        log_input: bool = True,
        n_krr_clfs: int = 1,
        no_posterior_collapse=True,
        mean_center: bool = False,
        unit_std: bool = False,
        frob_norm_source: bool = False,
        lib_size_norm: bool = False,
        n_jobs=1,
    ):
        """
        Create a SobolevAlignment instance.

        We allow the user to set all possible parameters from this API. However,
        the following choices were made in the manuscript presenting Sobolev Alignment (ref [1]):
        <ul>
        <li> source_batch_name and target_batch_name set to the experimental
        batch of cell lines and tumors respectively, allowing to use native
        scVI batch-effect correction within cell lines and tumors.
        <li> continuous_covariate_names set to None.
        <li> n_artificial_samples set to 10e7. In absence of large computing
        resource, a lower number (e.g. 10e6) could be used.
        <li> fit_vae, krr_approx and sample_artificial are set to True (running
        the complete pipeline) but playing with these
        three parameters allow to test different combination (e.g. kernel,
        penalization, number of artificial samples, ...)
        <li> n_samples_per_sample_batch set to 10e6. When not used on GPU,
        we recommend using 5*10e5.
        <li> save_mmap is set to /tmp/. This allows to save the model points
        in memory.
        <li> log_input and no_posterior_collapse are set to True.
        <li> All other parameters are False and n_krr_clfs is 1.
        </ul>

        Parameters
        ----------
        source_batch_name: str, default to None
            Name of the batch to use in scVI for batch-effect correction. If None,
            no batch-effect correction performed at the source-level.

        target_batch_name: str, default to None
            Name of the batch to use in scVI for batch-effect correction. If None,
            no batch-effect correction performed at the target-level.

        continuous_covariate_names: str, default to None
            Name of continuous covariate to use in scVI training. Will be used for
            both source and target.

        source_scvi_params
            Dictionary with scvi params for the source dataset. Must have three keys,
            each assigned to a dictionary of params: model, plan and train.

        target_scvi_params
            Dictionary with scvi params for the target dataset. Must have three keys,
            each assigned to a dictionary of params: model, plan and train.

        n_artificial_samples: int, default to 10e5
            Number of points to sample in both source and target scVI models in
            approximation. This corresponds to the number of "model points" used in
            the Kernel Ridge Regression step of source and target.

        n_samples_per_sample_batch: int, default to 10e6
            Number of samples per batch for sampling model points. This parameter
            does not affect the end-result, but can be used to alleviate memory
            issues in case of large n_artificial_samples.

        frac_save_artificial: float, default to 0.1
            Proportion of model points (artificial samples) to keep in memory. In
            case when several KRR models are trained this must be set to 1.
            <br/>
            Setting frac_save_artificial to 0.1 allows to compute the KRR approximation
            training error after the complete alignment.

        save_mmap: str, default to None
            Folder on disk to use for saving the model points (artificial data).
            This allows to limit the memory usage and therefore use larger KRR
            training data. If None, then artificial samples are kept in memory.
            <br/>
            This parameter does not affect the final prediction, simply the memory footprint.

        log_input: bool, default to True
            Whether model points (artificial samples) are log-transformed before
            being given as input to KRR. Log-transform usually increases approximation
            performance.

        n_krr_clfs: int, default to 1
            (Prototype) Number of KRR models to use. If larger than 1, the models
            prediction will be averaged.
            <br/>
            Experiments show no improvements when using more than one classifier.

        no_posterior_collapse: bool, default to True
            Whether posterior collapse should be avoided. If True, then scVI model
            is re-trained until no hidden neuron is collapsed. Every five iteration,
            one hidden neuron gets removed.

        mean_center: bool, default to False
            Whether model points (artificial samples) should be mean-centered before KRR.

        unit_std: bool, default to False
            Whether model points (artificial samples) should be standardized before KRR.

        frob_norm_source: bool, default to False
            In case when source and target data have a vastly different scale,
            frob_norm_source=True would correct the target model points (artificial samples)
            to have a median sample-wise Frobenius norm equal to the median
            sample-wise Frobenius norm of the source model points.

        lib_size_norm: bool, default to False
            Whether model points should be used with equal library size.

        n_jobs: int, default to 1
            Number of jobs to be used in parallelized operations. Default to  1.
            Setting n_jobs=-1 would use all the cpu cores.
        """
        # Save batch and continuous covariate names
        self.batch_name = {"source": source_batch_name, "target": target_batch_name}
        self.continuous_covariate_names = {"source": continuous_covariate_names, "target": continuous_covariate_names}

        # Save fitting parameters
        self._fit_params = {
            "n_artificial_samples": n_artificial_samples,
            "n_samples_per_sample_batch": n_samples_per_sample_batch,
            "frac_save_artificial": frac_save_artificial,
            "save_mmap": save_mmap,
            "log_input": log_input,
            "n_krr_clfs": n_krr_clfs,
            "no_posterior_collapse": no_posterior_collapse,
            "mean_center": mean_center,
            "unit_std": unit_std,
            "frob_norm_source": frob_norm_source,
            "lib_size_norm": lib_size_norm,
        }

        # scVI params
        self.scvi_params = {
            "source": source_scvi_params if source_scvi_params is not None else self.default_scvi_params,
            "target": target_scvi_params if target_scvi_params is not None else self.default_scvi_params,
        }
        for x in self.scvi_params:
            if "model" not in self.scvi_params[x]:
                self.scvi_params[x]["model"] = {}
            if "n_latent" not in self.scvi_params[x]["model"]:
                self.scvi_params[x]["model"]["n_latent"] = 10

        # KRR params
        self.krr_params = {
            "source": source_krr_params if source_krr_params is not None else {"method": "falkon"},
            "target": target_krr_params if target_krr_params is not None else {"method": "falkon"},
        }
        self._check_same_kernel()  # Check whether source and target have the same kernel
        self.scaler_ = {}

        # Create scVI models
        self.n_jobs = n_jobs

        # Initialize some values
        self._frob_norm_param = None

    def fit(
        self,
        X_source: AnnData,
        X_target: AnnData,
        fit_vae: bool = True,
        krr_approx: bool = True,
        sample_artificial: bool = True,
    ):
        """
        Run complete Sobolev Alignment workflow between a source (e.g. cell line) and a target (e.g. tumor) dataset.

        Source and target data should be passed as AnnData and potential batch names
        (source_batch_name, target_batch_name) should be part of the "obs" element
        of X_source and X_target.

        Parameters
        ----------
        X_source: AnnData
            Source data.

        X_target: AnnData
            Target data.

        fit_vae: bool, default to True
            Whether a scVI model (VAE) should be trained. If pre-trained VAEs are
            available, setting the scvi_models to these models and using
            fit_vae=False would allow to directly use these models.

        krr_approx: bool, default to True
            Whether the KRR approximation should be performed for source and target
            scVI models.

        sample_artificial: bool, default to True
            Whether model points should be sampled. In the case when artificial
            samples have already been sampled and saved, setting sample_artificial=False
            allows to use these points without need for re-sampling.

        Returns
        -------
        self: fitted Sobolev Alignment instance.

        """
        # Save data
        self.training_data = {"source": X_source, "target": X_target}

        # Train VAE
        if fit_vae:
            self._train_scvi_modules(no_posterior_collapse=self._fit_params["no_posterior_collapse"])

        # Sample artificial points
        if sample_artificial:
            self.lib_size = self._compute_batch_library_size()
            self.mean_center = self._fit_params["mean_center"]
            self.unit_std = self._fit_params["unit_std"]
            self.artificial_samples_ = {}
            self.artificial_embeddings_ = {}

        # Approximate KRR
        if krr_approx or sample_artificial:
            self.lib_size = self._compute_batch_library_size()
            self.approximate_krr_regressions_ = {}
            for data_source in ["source", "target"]:
                self._train_krr(
                    data_source=data_source,
                    n_artificial_samples=self._fit_params["n_artificial_samples"],
                    sample_artificial=sample_artificial,
                    krr_approx=krr_approx,
                    save_mmap=self._fit_params["save_mmap"],
                    log_input=self._fit_params["log_input"],
                    n_samples_per_sample_batch=self._fit_params["n_samples_per_sample_batch"],
                    frac_save_artificial=self._fit_params["frac_save_artificial"],
                    n_krr_clfs=self._fit_params["n_krr_clfs"],
                    mean_center=self.mean_center,
                    unit_std=self.unit_std,
                    frob_norm_source=self._fit_params["frob_norm_source"],
                    lib_size_norm=self._fit_params["lib_size_norm"],
                )

        # Comparison and alignment
        if krr_approx:
            self._compare_approximated_encoders()
            self._compute_principal_vectors()

        return self

    def _train_krr(
        self,
        data_source: str,
        n_artificial_samples: int,
        sample_artificial: bool = True,
        krr_approx: bool = True,
        save_mmap: str = None,
        log_input: bool = True,
        n_samples_per_sample_batch: int = 10**5,
        frac_save_artificial: float = 0.1,
        n_krr_clfs: int = 1,
        mean_center: bool = False,
        unit_std: bool = False,
        frob_norm_source: bool = False,
        lib_size_norm: bool = False,
    ):
        if n_krr_clfs == 1:
            self.approximate_krr_regressions_[data_source] = self._train_one_krr(
                data_source=data_source,
                n_artificial_samples=n_artificial_samples,
                sample_artificial=sample_artificial,
                krr_approx=krr_approx,
                save_mmap=save_mmap,
                log_input=log_input,
                n_samples_per_sample_batch=n_samples_per_sample_batch,
                frac_save_artificial=frac_save_artificial,
                mean_center=mean_center,
                unit_std=unit_std,
                frob_norm_source=frob_norm_source,
                lib_size_norm=lib_size_norm,
            )
            return True

        elif n_krr_clfs > 1:
            self.approximate_krr_regressions_[data_source] = MultiKRRApprox()
            for _ in range(n_krr_clfs):
                krr_approx = self._train_one_krr(
                    data_source=data_source,
                    n_artificial_samples=n_artificial_samples,
                    sample_artificial=sample_artificial,
                    krr_approx=krr_approx,
                    save_mmap=save_mmap,
                    log_input=log_input,
                    n_samples_per_sample_batch=n_samples_per_sample_batch,
                    frac_save_artificial=frac_save_artificial,
                    frob_norm_source=frob_norm_source,
                )
                self.approximate_krr_regressions_[data_source].add_clf(krr_approx)

            self.approximate_krr_regressions_[data_source].process_clfs()
            return True

    def _train_one_krr(
        self,
        data_source: str,
        n_artificial_samples: int,
        sample_artificial: bool = True,
        krr_approx: bool = True,
        save_mmap: str = None,
        log_input: bool = True,
        n_samples_per_sample_batch: int = 10**5,
        frac_save_artificial: float = 0.1,
        mean_center: bool = False,
        unit_std: bool = False,
        frob_norm_source: bool = False,
        lib_size_norm: bool = False,
    ):
        # Generate samples (decoder)
        if sample_artificial:
            artificial_samples, artificial_batches, artificial_covariates = self._generate_artificial_samples(
                data_source=data_source,
                n_artificial_samples=n_artificial_samples,
                large_batch_size=n_samples_per_sample_batch,
                save_mmap=save_mmap,
            )

            # Compute embeddings (encoder)
            artificial_embeddings = self._embed_artificial_samples(
                artificial_samples=artificial_samples,
                artificial_batches=artificial_batches,
                artificial_covariates=artificial_covariates,
                data_source=data_source,
                large_batch_size=n_samples_per_sample_batch,
            )
            gc.collect()

            # If artificial samples must be normalized for library size
            if lib_size_norm:
                artificial_samples = self._correct_artificial_samples_lib_size(
                    artificial_samples=artificial_samples,
                    artificial_batches=artificial_batches,
                    artificial_covariates=artificial_covariates,
                    data_source=data_source,
                    large_batch_size=n_samples_per_sample_batch,
                )
            del artificial_batches, artificial_covariates
            gc.collect()

            # Store in memmap
            artificial_samples = self._memmap_log_processing(
                data_source=data_source,
                artificial_samples=artificial_samples,
                artificial_embeddings=artificial_embeddings,
                save_mmap=save_mmap,
                log_input=log_input,
                mean_center=mean_center,
                unit_std=unit_std,
                frob_norm_source=frob_norm_source,
            )
        else:
            artificial_samples = self.artificial_samples_[data_source]
            artificial_embeddings = self.artificial_embeddings_[data_source]

        # KRR approx
        if krr_approx:
            krr_approx = self._approximate_encoders(
                data_source=data_source,
                artificial_samples=artificial_samples,
                artificial_embeddings=artificial_embeddings,
            )

        # Subsample the artificial sample saved
        if sample_artificial:
            n_save = int(frac_save_artificial * n_artificial_samples)
            subsampled_idx = np.random.choice(a=np.arange(n_artificial_samples), size=n_save, replace=False)
            self.artificial_samples_[data_source] = artificial_samples[subsampled_idx]
            del artificial_samples
            # Remove data in memmap
            if save_mmap is not None:
                os.remove(f"{save_mmap}/{data_source}_artificial_input.npy")
            self.artificial_embeddings_[data_source] = artificial_embeddings[subsampled_idx]
            # Remove data in memmap
            if save_mmap is not None:
                os.remove(f"{save_mmap}/{data_source}_artificial_embedding.npy")
            del artificial_embeddings
            gc.collect()
            torch.cuda.empty_cache()

        return krr_approx

    def _train_scvi_modules(self, no_posterior_collapse=False):
        """Train the scVI models based on data given and specifications."""
        self.scvi_models = {}
        self.scvi_batch_keys_ = {}

        for x in ["source", "target"]:
            self.training_data[x].layers["counts"] = self.training_data[x].X.copy()

            scvi.model.SCVI.setup_anndata(
                self.training_data[x],
                layer="counts",
                batch_key=self.batch_name[x],
                continuous_covariate_keys=self.continuous_covariate_names[x],
            )

            # Change covariates to float
            if self.continuous_covariate_names[x] is not None:
                for cov in self.continuous_covariate_names[x]:
                    self.training_data[x].obs[cov] = self.training_data[x].obs[cov].astype(np.float64)

            latent_variable_variance = np.zeros(1)
            save_iter = 0
            while np.any(latent_variable_variance < 0.2):
                logging.info(f"START TRAINING {x} model number {save_iter}")
                try:
                    self.scvi_models[x] = scvi.model.SCVI(self.training_data[x], **self.scvi_params[x]["model"])
                    self.scvi_models[x].train(plan_kwargs=self.scvi_params[x]["plan"], **self.scvi_params[x]["train"])
                except ValueError as err:
                    logging.error("\n SCVI TRAINING ERROR: \n %s \n\n\n\n" % (err))
                    latent_variable_variance = np.zeros(1)
                    continue

                if not no_posterior_collapse:
                    break
                else:
                    embedding = self.scvi_models[x].get_latent_representation()
                    latent_variable_variance = np.var(embedding, axis=0)
                    save_iter += 1

                if save_iter > 0 and save_iter % 5 == 0:
                    logging.info("\t SCVI: REMOVE ONE LATENT VARIABLE TO AVOID POSTERIOR COLLAPSE")
                    self.scvi_params[x]["model"]["n_latent"] = self.scvi_params[x]["model"]["n_latent"] - 1

            # Log batch key (used in data generation).
            if self.batch_name[x] is not None:
                dict_batch = (
                    self.training_data[x]
                    .obs[[self.batch_name[x], "_scvi_batch"]]
                    .reset_index(drop=True)
                    .drop_duplicates()
                )
                self.scvi_batch_keys_[x] = dict_batch.set_index(self.batch_name[x]).to_dict()["_scvi_batch"]
            else:
                self.scvi_batch_keys_[x] = None

        return True

    def _generate_artificial_samples(
        self, data_source: str, n_artificial_samples: int, large_batch_size: int = 10**5, save_mmap: str = None
    ):
        """
        Generate artificial samples for one model.

        Sample from the normal distribution associated to the latent space (for
        either source or target VAE model),
        generate some new data and process to recompute a new latent.

        Parameters
        ----------
        n_artificial_samples
            Number of artificial samples to produce for source and for target.

        Returns
        -------
        artificial_data: dict
            Dictionary containing the generated data for both source and target
        """
        batch_sizes = [large_batch_size] * (n_artificial_samples // large_batch_size) + [
            n_artificial_samples % large_batch_size
        ]
        batch_sizes = [x for x in batch_sizes if x > 0]
        _generated_data = [self._generate_artificial_samples_batch(batch, data_source) for batch in batch_sizes]
        _generated_data = list(zip(*_generated_data))
        artificial_samples = np.concatenate(_generated_data[0])
        artificial_batches_ = np.concatenate(_generated_data[1]) if self.batch_name[data_source] is not None else None
        artificial_covariates_ = (
            pd.concat(_generated_data[2]) if self.continuous_covariate_names[data_source] is not None else None
        )
        del _generated_data
        gc.collect()

        if save_mmap is not None and type(save_mmap) == str:
            np.save(open(f"{save_mmap}/{data_source}_artificial_input.npy", "wb"), artificial_samples)
            artificial_samples = np.load(f"{save_mmap}/{data_source}_artificial_input.npy", mmap_mode="r")
            gc.collect()

        return artificial_samples, artificial_batches_, artificial_covariates_

    def _generate_artificial_samples_batch(self, n_artificial_samples: int, data_source: str):
        artificial_batches = self._sample_batches(n_artificial_samples=n_artificial_samples, data=data_source)
        artificial_covariates = self._sample_covariates(n_artificial_samples=n_artificial_samples, data=data_source)
        artificial_samples = parallel_generate_samples(
            sample_size=n_artificial_samples,
            batch_names=artificial_batches,
            covariates_values=artificial_covariates,
            lib_size=self.lib_size[data_source],
            model=self.scvi_models[data_source],
            return_dist=False,
            batch_size=min(10**4, n_artificial_samples),
            batch_key_dict=self.scvi_batch_keys_[data_source],
            n_jobs=self.n_jobs,
        )

        non_zero_samples = torch.where(torch.sum(artificial_samples, axis=1) > 0)
        artificial_samples = artificial_samples[non_zero_samples]
        if artificial_covariates is not None:
            artificial_covariates = artificial_covariates.iloc[non_zero_samples]
        if artificial_batches is not None:
            artificial_batches = artificial_batches[non_zero_samples]
        gc.collect()

        return artificial_samples, artificial_batches, artificial_covariates

    def _compute_batch_library_size(self):
        if self.batch_name["source"] is None or self.batch_name["target"] is None:
            return {x: np.sum(self.training_data[x].X, axis=1).astype(float) for x in self.training_data}

        unique_batches = {x: np.unique(self.training_data[x].obs[self.batch_name[x]]) for x in self.training_data}

        return {
            x: {
                str(b): np.sum(
                    self.training_data[x][self.training_data[x].obs[self.batch_name[x]] == b].X, axis=1
                ).astype(float)
                for b in unique_batches[x]
            }
            for x in self.training_data
        }

    def _check_same_kernel(self):
        """Verify that same kernel is used for source and kernel KRR."""
        if "kernel" in self.krr_params["source"] or "kernel" in self.krr_params["target"]:
            assert self.krr_params["source"]["kernel"] == self.krr_params["target"]["kernel"]
        if "kernel_params" in self.krr_params["source"] or "kernel_params" in self.krr_params["target"]:
            assert self.krr_params["source"]["kernel_params"] == self.krr_params["target"]["kernel_params"]

    def _sample_batches(self, n_artificial_samples, data):
        """Sample batches for either source or target."""
        if self.batch_name[data] is None:
            return None

        return np.random.choice(
            self.training_data[data].obs[self.batch_name[data]].values, size=int(n_artificial_samples)
        )

    def _sample_covariates(self, n_artificial_samples, data):
        """Sample batches for either source or target."""
        if self.continuous_covariate_names[data] is None:
            return None

        return (
            self.training_data[data]
            .obs[self.continuous_covariate_names[data]]
            .sample(n_artificial_samples, replace=True)
        )

    def _embed_artificial_samples(
        self, artificial_samples, artificial_batches, artificial_covariates, data_source: str, large_batch_size=10**5
    ):
        # Divide in batches
        n_artificial_samples = artificial_samples.shape[0]
        batch_sizes = [large_batch_size] * (n_artificial_samples // large_batch_size) + [
            n_artificial_samples % large_batch_size
        ]
        batch_sizes = [0] + list(np.cumsum([x for x in batch_sizes if x > 0]))
        batch_start = batch_sizes[:-1]
        batch_end = batch_sizes[1:]

        # Format artificial samples to be fed into scVI.
        embedding = []
        for start, end in zip(batch_start, batch_end):
            x_train = artificial_samples[start:end]
            train_obs = pd.DataFrame(
                np.array(artificial_batches[start:end]) if artificial_batches is not None else [],
                columns=[self.batch_name[data_source]] if artificial_batches is not None else [],
                index=np.arange(end - start),
            )
            if artificial_covariates is not None:
                train_obs = pd.concat(
                    [train_obs, artificial_covariates.iloc[start:end].reset_index(drop=True)], ignore_index=True, axis=1
                )
                train_obs.columns = [self.batch_name[data_source], *self.continuous_covariate_names[data_source]]

            x_train_an = AnnData(x_train, obs=train_obs)
            x_train_an.layers["counts"] = x_train_an.X.copy()
            embedding.append(self.scvi_models[data_source].get_latent_representation(x_train_an))

        # Forward these formatted samples
        return np.concatenate(embedding)

    def _correct_artificial_samples_lib_size(
        self, artificial_samples, artificial_batches, artificial_covariates, data_source: str, large_batch_size=10**5
    ):
        """Correct for library size the artificial samples."""
        # Divide in batches
        n_artificial_samples = artificial_samples.shape[0]
        batch_sizes = [large_batch_size] * (n_artificial_samples // large_batch_size) + [
            n_artificial_samples % large_batch_size
        ]
        batch_sizes = [0] + list(np.cumsum([x for x in batch_sizes if x > 0]))
        batch_start = batch_sizes[:-1]
        batch_end = batch_sizes[1:]

        # Format artificial samples to be fed into scVI.
        artificial_samples = [artificial_samples[start:end] for start, end in zip(batch_start, batch_end)]
        for idx, (x_train, start, end) in enumerate(zip(artificial_samples, batch_start, batch_end)):
            train_obs = pd.DataFrame(
                np.array(artificial_batches[start:end]),
                columns=[self.batch_name[data_source]],
                index=np.arange(end - start),
            )
            if artificial_covariates is not None:
                train_obs = pd.concat(
                    [train_obs, artificial_covariates.iloc[start:end].reset_index(drop=True)], ignore_index=True, axis=1
                )
                train_obs.columns = [self.batch_name[data_source], *self.continuous_covariate_names[data_source]]

            x_train_an = AnnData(x_train, obs=train_obs)
            x_train_an.layers["counts"] = x_train_an.X.copy()
            artificial_samples[idx] = self.scvi_models[data_source].get_normalized_expression(
                x_train_an, return_numpy=True, library_size=DEFAULT_LIB_SIZE
            )

        artificial_samples = np.concatenate(artificial_samples)

        return artificial_samples

    def _memmap_log_processing(
        self,
        data_source: str,
        artificial_samples,
        artificial_embeddings,
        save_mmap: str = None,
        log_input: bool = False,
        mean_center: bool = False,
        unit_std: bool = False,
        frob_norm_source: bool = False,
    ):
        # Save embedding
        if save_mmap is not None and type(save_mmap) == str:
            self._save_mmap = save_mmap
            self._memmap_embedding(
                data_source=data_source, artificial_embeddings=artificial_embeddings, save_mmap=save_mmap
            )

        self.krr_log_input_ = log_input
        if log_input:
            artificial_samples = np.log10(artificial_samples + 1)

            # Standard Scaler
            scaler_ = StandardScaler(with_mean=mean_center, with_std=unit_std)
            artificial_samples = scaler_.fit_transform(np.array(artificial_samples))

            # Frobenius norm scaling
            artificial_samples = self._frobenius_normalisation(data_source, artificial_samples, frob_norm_source)

            if save_mmap is not None and type(save_mmap) == str:
                # Re-save
                np.save(open(f"{save_mmap}/{data_source}_artificial_input.npy", "wb"), artificial_samples)
                artificial_samples = np.load(f"{save_mmap}/{data_source}_artificial_input.npy", mmap_mode="r")
                gc.collect()

            else:
                pass

        else:
            # Frobenius norm scaling
            artificial_samples = self._frobenius_normalisation(data_source, artificial_samples, frob_norm_source)

        return artificial_samples

    def _frobenius_normalisation(self, data_source, artificial_samples, frob_norm_source):
        # Normalise to same Frobenius norm per sample
        if frob_norm_source:
            if data_source == "source":
                self._frob_norm_param = np.mean(np.linalg.norm(artificial_samples, axis=1))
            else:
                frob_norm = np.mean(np.linalg.norm(artificial_samples, axis=1))
                artificial_samples = artificial_samples * self._frob_norm_param / frob_norm
        else:
            pass

        return artificial_samples

    def _memmap_embedding(self, data_source, artificial_embeddings, save_mmap):
        np.save(open(f"{save_mmap}/{data_source}_artificial_embedding.npy", "wb"), artificial_embeddings)
        artificial_embeddings = np.load(f"{save_mmap}/{data_source}_artificial_embedding.npy", mmap_mode="r")
        gc.collect()

        return artificial_embeddings

    def _approximate_encoders(self, data_source: str, artificial_samples, artificial_embeddings):
        """Approximate the encoder by a KRR regression."""
        krr_approx = KRRApprox(**self.krr_params[data_source])

        krr_approx.fit(torch.from_numpy(artificial_samples), torch.from_numpy(artificial_embeddings))

        return krr_approx

    def _compare_approximated_encoders(self):
        self.M_X = self._compute_cosine_sim_intra_dataset("source")
        self.M_Y = self._compute_cosine_sim_intra_dataset("target")
        self.M_XY = self._compute_cross_cosine_sim()

        self.sqrt_inv_M_X_ = mat_inv_sqrt(self.M_X)
        self.sqrt_inv_M_Y_ = mat_inv_sqrt(self.M_Y)
        self.sqrt_inv_matrices_ = {"source": self.sqrt_inv_M_X_, "target": self.sqrt_inv_M_Y_}
        self.cosine_sim = self.sqrt_inv_M_X_.dot(self.M_XY).dot(self.sqrt_inv_M_Y_)

    def _compute_cosine_sim_intra_dataset(self, data: str):
        """
        Compute M_X if data='source', or M_Y if data='target'.

        :param data:
        :return:
        """
        krr_clf = self.approximate_krr_regressions_[data]
        K = krr_clf.kernel_(krr_clf.anchors(), krr_clf.anchors())
        K = torch.Tensor(K)
        return krr_clf.sample_weights_.T.matmul(K).matmul(krr_clf.sample_weights_)

    def _compute_cross_cosine_sim(self):
        K_XY = self.approximate_krr_regressions_["target"].kernel_(
            self.approximate_krr_regressions_["source"].anchors(), self.approximate_krr_regressions_["target"].anchors()
        )
        K_XY = torch.Tensor(K_XY)
        return (
            self.approximate_krr_regressions_["source"]
            .sample_weights_.T.matmul(K_XY)
            .matmul(self.approximate_krr_regressions_["target"].sample_weights_)
        )

    def _compute_principal_vectors(self, all_PVs=False):
        """
        Compute principal vectors by SVD of cosine similarity.

        All_PVs indicate whether the data source with the most PVs should be reduced to the
        number of PVs of the smallest data-source.
        Example: source has 10 factors, target 13. all_PVs=True would yield 13 target PVs,
        all_PVs=False would yield 10.
        """
        cosine_svd = np.linalg.svd(self.cosine_sim, full_matrices=all_PVs)
        self.principal_angles = cosine_svd[1]
        self.untransformed_rotations_ = {"source": cosine_svd[0], "target": cosine_svd[2].T}
        self.principal_vectors_coef_ = {
            x: self.untransformed_rotations_[x]
            .T.dot(self.sqrt_inv_matrices_[x])
            .dot(self.approximate_krr_regressions_[x].sample_weights_.T.detach().numpy())
            for x in self.untransformed_rotations_
        }

    def compute_consensus_features(self, X_input: dict, n_similar_pv: int, fit: bool = True, return_anndata=False):
        """
        Project data on interpolated consensus features.

        Project the data on interpolated features, i.e., a linear combination of source and target SPVs which
        best balances the effect of source and target data.

        Parameters
        ----------
        X_input: dict
            Dictionary of data (AnnData) to project. Two keys are needed: 'source' and 'target'.
        n_similar_pv: int
            Number of top SPVs to project the data on.
        fit: bool, default to True
            Whether the interpolated times must be computed. If False, will use previously computed times,
            but will return an error if not previously fitted.
        return_anndata: bool, default to False
            Whether the projected consensus features must be formatted as an AnnData with overlapping
            indices in obs. This allows downstream analysis. By default, return a DataFrame.

        Returns
        -------
        interpolated_proj_df: pd.DataFrame or sc.AnnData
            DataFrame or AnnData of concatenated source and target samples after projection on consensus features.
        """
        X_data_log = {
            data_source: self._frobenius_normalisation(
                data_source, torch.log10(torch.Tensor(X_input[data_source].X + 1)), frob_norm_source=True
            )
            for data_source in ["source", "target"]
        }

        # Project data on KRR directions
        krr_projections = {
            pv_data_source: {
                proj_data_source: self.approximate_krr_regressions_[pv_data_source]
                .transform(X_data_log[proj_data_source])
                .detach()
                .numpy()
                for proj_data_source in ["source", "target"]
            }
            for pv_data_source in ["source", "target"]
        }

        # Rotate KRR directions to obtain PVs
        pv_projections = {}
        for pv_data_source in krr_projections:
            pv_projections[pv_data_source] = {}
            for proj_data_source in krr_projections[pv_data_source]:
                rotated_proj = self.untransformed_rotations_[pv_data_source].T
                rotated_proj = rotated_proj.dot(self.sqrt_inv_matrices_[pv_data_source])
                rotated_proj = rotated_proj.dot(krr_projections[pv_data_source][proj_data_source].T).T

                pv_projections[pv_data_source][proj_data_source] = rotated_proj
        del rotated_proj

        # Mean-center projection data on the PV
        pv_projections = {
            pv_data_source: {
                proj_data_source: StandardScaler(with_mean=True, with_std=False).fit_transform(
                    pv_projections[pv_data_source][proj_data_source]
                )
                for proj_data_source in ["source", "target"]
            }
            for pv_data_source in ["source", "target"]
        }

        # Compute optimal interpolation point
        if fit:
            self.n_similar_pv = n_similar_pv
            self.optimal_interpolation_step_ = {
                PV_number: compute_optimal_tau(
                    PV_number, pv_projections, np.arccos(self.principal_angles), n_interpolation=100
                )
                for PV_number in range(self.n_similar_pv)
            }

        # Project on optimal interpolation time
        interpolated_proj_df = {
            PV_number: np.concatenate(
                list(
                    project_on_interpolate_PV(
                        np.arccos(self.principal_angles)[PV_number], PV_number, optimal_step, pv_projections
                    )
                )
            )
            for PV_number, optimal_step in self.optimal_interpolation_step_.items()
        }
        interpolated_proj_df = pd.DataFrame(interpolated_proj_df)

        if return_anndata:
            interpolated_proj_an = sc.concat([X_input["source"], X_input["target"]])
            interpolated_proj_an.obs["data_source"] = ["source"] * X_input["source"].shape[0] + ["target"] * X_input[
                "target"
            ].shape[0]
            interpolated_proj_an.obsm["X_sobolev_alignment"] = np.array(interpolated_proj_df)
            return interpolated_proj_an
        else:
            return interpolated_proj_df

    def scvi_model_selection(
        self,
        X_source: AnnData,
        X_target: AnnData,
        source_batch_name: str = None,
        target_batch_name: str = None,
        model=scvi.model.SCVI,
        space: dict = DEFAULT_HYPEROPT_SPACE,
        max_eval: int = 100,
        test_size: float = 0.1,
    ):
        """
        Hyperparameter selection for scVI models.

        Routine to perform Bayesian hyper-parameter optimisation for scVI model (source and target).
        Can be called prior to fit. Best parameters will be saved in self.scvi_params

        Parameters
        ----------
        X_source: AnnData
            Source dataset.
        X_target: AnnData
            Target dataset.
        source_batch_name: str, default to None
            Batch key to use in scVI for the source dataset. If None, no native
            batch-effect correction performed in source scVI.
        target_batch_name: str, default to None
            Batch key to use in scVI for the target dataset. If None, no native
            batch-effect correction performed in target scVI.
        model: default to scvi.model.SCVI
            scvi-tools model to be used in the analysis.
        space: dict, default to DEFAULT_HYPEROPT_SPACE
            Hyper-parameter space to be used in Bayesian Optimisation. Default is
            provided in sobolev_alignment.scvi_model_search.
        max_eval: int, default to 100
            Number of iterations in the Bayesian optimisation procedures, i.e., number
            of models assessed.
        test_size: float, default to 0.1
            Proportion of samples (cells) to be taken inside the test data.

        Returns
        -------
            SobolevAlignment instance.
        """
        data = {"source": X_source, "target": X_target}
        batch_names = {"source": source_batch_name, "target": target_batch_name}

        self.scvi_params = {}
        self.scvi_hyperopt_performances_df = {}
        for x in data.keys():
            print("START MODEL SELECTION FOR %s" % (x), flush=True)
            h = model_selection(
                data_an=data[x],
                batch_key=batch_names[x],
                model=model,
                space=space,
                max_eval=max_eval,
                test_size=test_size,
                save=None,
            )

            self.scvi_params[x] = {
                "model": {k: h[0][k] for k in h[0] if k in SCVI_MODEL_PARAMS},
                "plan": {k: h[0][k] for k in h[0] if k in SCVI_PLAN_PARAMS},
                "train": {k: h[0][k] for k in h[0] if k in SCVI_TRAIN_PARAMS},
            }
            self.scvi_hyperopt_performances_df[x] = h[1]

        return self

    def krr_model_selection(
        self, X_source: AnnData, X_target: AnnData, M: int = 1000, same_model_alignment_thresh: float = 0.9
    ):
        """
        Hyper-parameters selection for KRR.

        Routine to perform Bayesian hyper-parameter optimisation for scVI model (source and target).
        Can be called prior to fit. Best parameters will be saved in self.scvi_params

        Parameters
        ----------
        X_source: AnnData
            Source dataset.
        X_target: AnnData
            Target dataset.
        M: int, default to 1000
            Number of anchor points to use. Larger values of M leads to a better
            approximation of the latent factors, but come at the price of a higher
            computational time and memory.
        same_model_alignment_thresh: float, default to 0.9
            Minimum top principal angles used during same-model alignment, i.e., when
            source or target models are aligned to themselves.

        Returns
        -------
            SobolevAlignment instance.
        """
        # Log input if required
        X_input = {"source": X_source.X, "target": X_target.X}
        if self._fit_params["log_input"]:
            X_input = {k: np.log10(X_input[k] + 1) for k in X_input}

        # Compute sigma after re-scaling data (if required)
        if self._fit_params["frob_norm_source"]:
            X_input["source"] = self._frobenius_normalisation("source", X_input["source"], frob_norm_source=True)
            X_input["target"] = self._frobenius_normalisation("target", X_input["target"], frob_norm_source=True)
        source_target_distance = np.power(pairwise_distances(X_input["source"], X_input["target"]), 2)
        sigma = np.sqrt(np.mean(source_target_distance) / np.log(2))
        print("OPTIMAL SIGMA: %1.2f" % (sigma))

        # Select nu
        nu_selection_error_df = model_selection_nu(
            X_source=X_source,
            X_target=X_target,
            sobolev_alignment_clf=self,
            sigma=sigma,
            M=M,
            test_error_size=int(0.1 * self._fit_params["n_artificial_samples"]),
        )
        optimal_krr_nu = float(nu_selection_error_df.index[0].split("$nu:")[-1])
        print("OPTIMAL NU: %1.2f" % (optimal_krr_nu))

        # Select penalization
        self.penalization_principal_angles_df_ = {}
        for data_source in ["source", "target"]:
            self.krr_params[data_source]["kernel_params"] = {"sigma": sigma, "nu": optimal_krr_nu}
            self.penalization_principal_angles_df_[data_source] = model_alignment_penalization(
                X_data=X_source if data_source == "source" else X_target,
                data_source=data_source,
                sobolev_alignment_clf=self,
                sigma=sigma,
                optimal_nu=optimal_krr_nu,
                M=M,
            )

            self.penalization_principal_angles_df_[data_source] = (
                pd.DataFrame(self.penalization_principal_angles_df_[data_source]).iloc[:1] > same_model_alignment_thresh
            ).T

            pen = re.findall(
                r"penalization_([0-9.e\-]*)",
                self.penalization_principal_angles_df_[data_source]
                .loc[self.penalization_principal_angles_df_[data_source][0], 0]
                .index[0],
            )
            if len(pen) == 0:
                self.krr_params[data_source]["penalization"] = 1e3
            else:
                self.krr_params[data_source]["penalization"] = float(pen[0])
            del pen

            self.krr_params[data_source]["M"] = M

        return self.krr_params

    def save(self, folder: str = ".", with_krr: bool = True, with_model: bool = True):
        """Save Sobolev Alignment model."""
        if not os.path.exists(folder) and not os.path.isdir(folder):
            os.mkdir(folder)

        dump(self.batch_name, open("%s/batch_name.pkl" % (folder), "wb"))
        dump(self.continuous_covariate_names, open("%s/continuous_covariate_names.pkl" % (folder), "wb"))

        # Dump scVI models
        if with_model:
            for x in self.scvi_models:
                dump(self.scvi_models[x], open(f"{folder}/scvi_model_{x}.pkl", "wb"))
                self.scvi_models[x].save(f"{folder}/scvi_model_{x}", save_anndata=True)
                dump(self.scvi_batch_keys_[x], open(f"{folder}/scvi_model_key_dict_{x}.pkl", "wb"))

        # Dump the KRR:
        if not with_krr:
            return True

        for x in self.approximate_krr_regressions_:
            self.approximate_krr_regressions_[x].save(f"{folder}/krr_approx_{x}")

        # Save params
        pd.DataFrame(self.krr_params).to_csv("%s/krr_params.csv" % (folder))
        dump(self.krr_params, open("%s/krr_params.pkl" % (folder), "wb"))

        for param_t in ["model", "plan", "train"]:
            df = pd.DataFrame([self.scvi_params[x][param_t] for x in ["source", "target"]])
            df.to_csv(f"{folder}/scvi_params_{param_t}.csv")
        dump(self.scvi_params, open("%s/scvi_params.pkl" % (folder), "wb"))

        pd.DataFrame(self._fit_params, index=["params"]).to_csv("%s/fit_params.csv" % (folder))
        dump(self._fit_params, open("%s/fit_params.pkl" % (folder), "wb"))

        # Save results
        results_elements = {
            "alignment_M_X": self.M_X,
            "alignment_M_Y": self.M_Y,
            "alignment_M_XY": self.M_XY,
            "alignment_cosine_sim": self.cosine_sim,
            "alignment_principal_angles": self.principal_angles,
        }
        for idx, element in results_elements.items():
            if type(element) is np.ndarray:
                np.savetxt(f"{folder}/{idx}.csv", element)
                np.save(open(f"{folder}/{idx}.npy", "wb"), element)
            elif type(element) is torch.Tensor:
                np.savetxt(f"{folder}/{idx}.csv", element.detach().numpy())
                torch.save(element, open(f"{folder}/{idx}.pt", "wb"))

        if self._frob_norm_param is not None:
            np.savetxt("%s/frob_norm_param.csv" % (folder), np.array([self._frob_norm_param]))

    def load(folder: str = ".", with_krr: bool = True, with_model: bool = True):
        """
        Load a Sobolev Alignment instance.

        Parameters
        ----------
        folder: str, default to '.'
            Folder path where the instance is located
        with_krr: bool, default to True
            Whether KRR approximations must be loaded.
        with_model: bool, default to True
            Whether scvi models (VAEs) must be loaded.

        Returns
        -------
        SobolevAlignment: instance saved at the folder location.
        """
        clf = SobolevAlignment()

        if "batch_name.pkl" in os.listdir(folder):
            clf.batch_name = load(open("%s/batch_name.pkl" % (folder), "rb"))
        if "continuous_covariate_names.pkl" in os.listdir(folder):
            clf.continuous_covariate_names = load(open("%s/continuous_covariate_names.pkl" % (folder), "rb"))

        if with_model:
            clf.scvi_models = {}
            clf.scvi_batch_keys_ = {}
            for x in ["source", "target"]:
                clf.scvi_models[x] = scvi.model.SCVI.load(f"{folder}/scvi_model_{x}")
                if "scvi_model_key_dict_%s.pkl" % (x) in os.listdir(folder):
                    clf.scvi_batch_keys_[x] = load(open(f"{folder}/scvi_model_key_dict_{x}.pkl", "rb"))
                else:
                    clf.scvi_batch_keys_[x] = None

        if with_krr:
            clf.approximate_krr_regressions_ = {}
            for x in ["source", "target"]:
                clf.approximate_krr_regressions_[x] = KRRApprox.load(f"{folder}/krr_approx_{x}/")

            # Load params
            clf.krr_params = load(open("%s/krr_params.pkl" % (folder), "rb"))
            clf.scvi_params = load(open("%s/scvi_params.pkl" % (folder), "rb"))
            if "fit_params.pkl" in os.listdir(folder):
                clf._fit_params = load(open("%s/fit_params.pkl" % (folder), "rb"))

            # Load results
            if "alignment_M_X.npy" in os.listdir(folder):
                clf.M_X = np.load("%s/alignment_M_X.npy" % (folder))
            elif "alignment_M_X.pt" in os.listdir(folder):
                clf.M_X = torch.load(open("%s/alignment_M_X.pt" % (folder), "rb"))

            if "alignment_M_Y.npy" in os.listdir(folder):
                clf.M_Y = np.load("%s/alignment_M_Y.npy" % (folder))
            elif "alignment_M_Y.pt" in os.listdir(folder):
                clf.M_Y = torch.load(open("%s/alignment_M_Y.pt" % (folder), "rb"))

            if "alignment_M_XY.npy" in os.listdir(folder):
                clf.M_XY = np.load("%s/alignment_M_XY.npy" % (folder))
            elif "alignment_M_XY.pt" in os.listdir(folder):
                clf.M_XY = torch.load(open("%s/alignment_M_XY.pt" % (folder), "rb"))

            if "alignment_cosine_sim.npy" in os.listdir(folder):
                clf.cosine_sim = np.load("%s/alignment_cosine_sim.npy" % (folder))
            elif "alignment_cosine_sim.pt" in os.listdir(folder):
                clf.cosine_sim = torch.load(open("%s/alignment_cosine_sim.pt" % (folder), "rb"))

            if "alignment_principal_angles.npy" in os.listdir(folder):
                clf.principal_angles = np.load("%s/alignment_principal_angles.npy" % (folder))
            elif "alignment_principal_angles.pt" in os.listdir(folder):
                clf.principal_angles = torch.load(open("%s/alignment_principal_angles.pt" % (folder), "rb"))

            clf.sqrt_inv_M_X_ = mat_inv_sqrt(clf.M_X)
            clf.sqrt_inv_M_Y_ = mat_inv_sqrt(clf.M_Y)
            clf.sqrt_inv_matrices_ = {"source": clf.sqrt_inv_M_X_, "target": clf.sqrt_inv_M_Y_}
            clf._compute_principal_vectors()

        if "frob_norm_param.csv" in os.listdir(folder):
            clf._frob_norm_param = np.loadtxt(open("%s/frob_norm_param.csv" % (folder)))

        return clf

    def plot_training_metrics(self, folder: str = "."):
        """Plot the different training metric for the source and target scVI modules."""
        if not os.path.exists(folder) and not os.path.isdir(folder):
            os.mkdir(folder)

        for x in self.scvi_models:
            for metric in self.scvi_models[x].history:
                plt.figure(figsize=(6, 4))
                plt.plot(self.scvi_models[x].history[metric])
                plt.xlabel("Epoch", fontsize=20, color="black")
                plt.ylabel(metric, fontsize=20, color="black")
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.tight_layout()
                plt.savefig(f"{folder}/{x}_model_train_{metric}.png", dpi=300)
                plt.show()

    def plot_cosine_similarity(self, folder: str = ".", absolute_cos: bool = False):
        """Plot cosine similarity."""
        if absolute_cos:
            sns.heatmap(np.abs(self.cosine_sim), cmap="seismic_r", center=0)
        else:
            sns.heatmap(self.cosine_sim, cmap="seismic_r", center=0)
        plt.xticks(fontsize=12, color="black")
        plt.yticks(fontsize=12, color="black")
        plt.xlabel("Tumor", fontsize=25, color="black")
        plt.ylabel("Cell lines", fontsize=25)
        plt.tight_layout()
        plt.savefig("{}/{}cosine_similarity.png".format(folder, "abs_" if absolute_cos else ""), dpi=300)
        plt.show()

    def compute_error(self, size=-1):
        """Compute error of the KRR approximation on the input (data used for VAE training) and used for KRR."""
        return {
            "source": self._compute_error_one_type("source", size=size),
            "target": self._compute_error_one_type("target", size=size),
        }

    def _compute_error_one_type(self, data_type, size=-1):
        # KRR error of input data
        latent = self.scvi_models[data_type].get_latent_representation()
        if self._fit_params["lib_size_norm"]:
            input_krr_pred = self.scvi_models[data_type].get_normalized_expression(
                return_numpy=True, library_size=DEFAULT_LIB_SIZE
            )
        else:
            input_krr_pred = self.training_data[data_type].X
        if self.krr_log_input_:
            input_krr_pred = np.log10(input_krr_pred + 1)

        if data_type == " target":
            input_krr_pred = self._frobenius_normalisation(data_type, input_krr_pred, self._frob_norm_param is not None)

        input_krr_pred = StandardScaler(with_mean=self.mean_center, with_std=self.unit_std).fit_transform(
            input_krr_pred
        )
        input_krr_pred = self.approximate_krr_regressions_[data_type].transform(torch.Tensor(input_krr_pred))
        input_spearman_corr = np.array([scipy.stats.spearmanr(x, y)[0] for x, y in zip(input_krr_pred.T, latent.T)])
        input_krr_diff = input_krr_pred - latent
        input_mean_square = torch.square(input_krr_diff)
        input_factor_mean_square = torch.mean(input_mean_square, axis=0)
        input_latent_mean_square = torch.mean(input_mean_square)
        input_factor_reconstruction_error = np.linalg.norm(input_krr_diff, axis=0) / np.linalg.norm(latent, axis=0)
        input_latent_reconstruction_error = np.linalg.norm(input_krr_diff) / np.linalg.norm(latent)
        del input_krr_pred, input_mean_square, input_krr_diff
        gc.collect()

        # KRR error of artificial data
        if size > 1:
            subsamples = np.random.choice(np.arange(self.artificial_samples_[data_type].shape[0]), size, replace=False)
        elif size <= 0:
            return {
                "factor": {
                    "MSE": {"input": input_factor_mean_square.detach().numpy()},
                    "reconstruction_error": {"input": input_factor_reconstruction_error},
                    "spearmanr": {"input": np.array(input_spearman_corr)},
                },
                "latent": {
                    "MSE": {"input": input_latent_mean_square.detach().numpy()},
                    "reconstruction_error": {"input": input_latent_reconstruction_error},
                    "spearmanr": {"input": np.mean(input_spearman_corr)},
                },
            }
        else:
            subsamples = np.arange(self.artificial_samples_[data_type].shape[0])
        training_krr_diff = self.approximate_krr_regressions_[data_type].transform(
            torch.Tensor(self.artificial_samples_[data_type][subsamples])
        )
        training_spearman_corr = np.array(
            [
                scipy.stats.spearmanr(x, y)[0]
                for x, y in zip(training_krr_diff.T, self.artificial_embeddings_[data_type][subsamples].T)
            ]
        )
        training_krr_diff = training_krr_diff - self.artificial_embeddings_[data_type][subsamples]
        training_krr_factor_reconstruction_error = np.linalg.norm(training_krr_diff, axis=0) / np.linalg.norm(
            self.artificial_embeddings_[data_type][subsamples], axis=0
        )
        training_krr_latent_reconstruction_error = np.linalg.norm(training_krr_diff) / np.linalg.norm(
            self.artificial_embeddings_[data_type][subsamples]
        )

        return {
            "factor": {
                "MSE": {
                    "input": input_factor_mean_square.detach().numpy(),
                    "artificial": torch.mean(torch.square(training_krr_diff), axis=0).detach().numpy(),
                },
                "reconstruction_error": {
                    "input": input_factor_reconstruction_error,
                    "artificial": training_krr_factor_reconstruction_error,
                },
                "spearmanr": {"input": np.array(input_spearman_corr), "artificial": np.array(training_spearman_corr)},
            },
            "latent": {
                "MSE": {
                    "input": input_latent_mean_square.detach().numpy(),
                    "artificial": torch.mean(torch.square(training_krr_diff)).detach().numpy(),
                },
                "reconstruction_error": {
                    "input": input_latent_reconstruction_error,
                    "artificial": training_krr_latent_reconstruction_error,
                },
                "spearmanr": {"input": np.mean(input_spearman_corr), "artificial": np.mean(training_spearman_corr)},
            },
        }

    def feature_analysis(self, max_order: int = 1, gene_names: list = None):
        """
        Launch feature analysis for a trained scVI model.

        Computes the gene contributions (feature weights) associated with the KRRs which approximate the
        latent factors and the SPVs. Technically, given the kernel machine which approximates a latent factor
        (KRR), this method computes the weights associated with the orthonormal basis in the Gaussian-kernel
        associated Sobolev space.

        Parameters
        ----------
        max_order: int, default to 1
            Order of the features to compute. 1 corresponds to linear features (genes), two to interaction terms.
        gene_names: list of str, default to None
            Names of the genes passed as input to Sobolev Alignment. <b>WARNING</b> Must be in the same order as
            the input to SobolevAlignment.fit
        """
        # Make kernel parameter
        if (
            "gamma" in self.krr_params["source"]["kernel_params"]
            and "gamma" in self.krr_params["target"]["kernel_params"]
        ):
            gamma_s = self.krr_params["source"]["kernel_params"]["gamma"]
            gamma_t = self.krr_params["target"]["kernel_params"]["gamma"]
        elif (
            "sigma" in self.krr_params["source"]["kernel_params"]
            and "sigma" in self.krr_params["target"]["kernel_params"]
        ):
            gamma_s = 1 / (2 * self.krr_params["source"]["kernel_params"]["sigma"] ** 2)
            gamma_t = 1 / (2 * self.krr_params["target"]["kernel_params"]["sigma"] ** 2)
        assert gamma_s == gamma_t
        self.gamma = gamma_s

        # Compute the sample offset (matrix O_X and O_Y)
        self.sample_offset = {
            x: _compute_offset(self.approximate_krr_regressions_[x].anchors(), self.gamma) for x in self.training_data
        }

        if gene_names is None:
            self.gene_names = self.training_data["source"].columns
        else:
            self.gene_names = gene_names

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.factor_level_feature_weights_df = {}
        for x in self.training_data:
            # Computes all the features of order d.
            basis_feature_weights_df = higher_order_contribution(
                d=max_order,
                data=self.approximate_krr_regressions_[x].anchors().cpu().detach().numpy(),
                sample_offset=self.sample_offset[x],
                gene_names=self.gene_names,
                gamma=self.gamma,
                n_jobs=self.n_jobs,
            )

            # Process feature weights.
            index = np.arange(self.approximate_krr_regressions_[x].sample_weights_.T.shape[0])
            columns = basis_feature_weights_df.columns
            values = self.approximate_krr_regressions_[x].sample_weights_.T.to(device)
            values = values.matmul(torch.Tensor(basis_feature_weights_df.values).to(device))
            self.factor_level_feature_weights_df[x] = pd.DataFrame(
                values.cpu().detach().numpy(), index=index, columns=columns
            )
            del basis_feature_weights_df
            gc.collect()

        # Compute SPV weights by rotating the factor-level weights.
        self.pv_level_feature_weights_df = {
            x: pd.DataFrame(
                self.untransformed_rotations_[x]
                .T.dot(self.sqrt_inv_matrices_[x])
                .dot(self.factor_level_feature_weights_df[x]),
                index=["PV %s" % (i) for i in range(self.untransformed_rotations_[x].shape[1])],
                columns=self.factor_level_feature_weights_df[x].columns,
            )
            for x in self.training_data
        }

    def sample_random_vector_(self, data_source, K):
        """Sample a vector randomly for either source or target."""
        n_samples = self.approximate_krr_regressions_[data_source].anchors().shape[0]
        n_factors = self.approximate_krr_regressions_[data_source].anchors().shape[1]

        # Random coefficients
        coefficients = torch.randn(n_samples, n_factors)

        # Random norms
        M = self.M_X if data_source == "source" else self.M_Y
        factor_norms = torch.FloatTensor(n_factors).uniform_(
            torch.sqrt(torch.min(torch.linalg.svd(M)[1])), torch.sqrt(torch.max(torch.linalg.svd(M)[1]))
        )

        # Gram-Schmidt
        for j in range(n_factors):
            for i in range(j):
                similarity = coefficients[:, i].matmul(K).matmul(coefficients[:, j])
                coefficients[:, j] = coefficients[:, j] - similarity * coefficients[:, i]
            # Normalise
            coefficients[:, j] = coefficients[:, j] / torch.sqrt(
                coefficients[:, j].matmul(K).matmul(coefficients[:, j])
            )

        # Correct for norm
        norm_vectors = torch.sqrt(torch.diag(coefficients.T.matmul(K).matmul(coefficients)))
        coefficients = coefficients / norm_vectors * factor_norms

        return coefficients

    def compute_random_direction_(self, K_X, K_Y, K_XY):
        """Sample randomly two vectors and compute cosine similarity."""
        # Random samples
        perm_source_sample_coef = self.sample_random_vector_("source", K_X)
        perm_target_sample_coef = self.sample_random_vector_("target", K_Y)

        # Computation of cosine similarity matrix
        M_XY_perm_uncorrected = perm_source_sample_coef.T.matmul(K_XY).matmul(perm_target_sample_coef)
        M_X_perm = perm_source_sample_coef.T.matmul(K_X).matmul(perm_source_sample_coef)
        M_Y_perm = perm_target_sample_coef.T.matmul(K_Y).matmul(perm_target_sample_coef)
        inv_M_X_perm = mat_inv_sqrt(M_X_perm)
        inv_M_Y_perm = mat_inv_sqrt(M_Y_perm)

        return np.linalg.svd(inv_M_X_perm.dot(M_XY_perm_uncorrected).dot(inv_M_Y_perm))[1]

    def null_model_similarity(self, n_iter=100, quantile=0.95, return_all=False, n_jobs=1):
        """Compute the null model for PV similarities."""
        # Compute similarity
        K_X = self.approximate_krr_regressions_["target"].kernel_(
            self.approximate_krr_regressions_["source"].anchors(), self.approximate_krr_regressions_["source"].anchors()
        )
        K_Y = self.approximate_krr_regressions_["target"].kernel_(
            self.approximate_krr_regressions_["target"].anchors(), self.approximate_krr_regressions_["target"].anchors()
        )
        K_XY = self.approximate_krr_regressions_["target"].kernel_(
            self.approximate_krr_regressions_["source"].anchors(), self.approximate_krr_regressions_["target"].anchors()
        )

        random_directions = Parallel(n_jobs=n_jobs, verbose=1, backend="threading")(
            delayed(self.compute_random_direction_)(K_X, K_Y, K_XY) for _ in range(n_iter)
        )

        if return_all:
            return np.array(random_directions)
        return np.quantile(np.array(random_directions)[:, 0], quantile)
