import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from sobolev_alignment import SobolevAlignment
from sobolev_alignment.krr_approx import FALKON_IMPORTED

n_samples = 500
n_samples_valid = 50
n_genes = 50
n_batches = 3
n_artificial_samples = 2000
n_latent = 5
frac_save_artificial = 0.1


@pytest.fixture(scope="module")
def falkon_import():
    return FALKON_IMPORTED


@pytest.fixture(scope="module")
def source_data():
    poisson_coef = np.random.randint(1, 25, size=n_genes)
    return np.concatenate([np.random.poisson(lam=l, size=n_samples).reshape(-1, 1) for l in poisson_coef], axis=1)


@pytest.fixture(scope="module")
def target_data():
    poisson_coef = np.random.randint(1, 25, size=n_genes)
    return np.concatenate([np.random.poisson(lam=l, size=n_samples).reshape(-1, 1) for l in poisson_coef], axis=1)


@pytest.fixture(scope="module")
def source_batch():
    return np.random.choice(np.arange(n_batches).astype(str), size=n_samples)


@pytest.fixture(scope="module")
def target_batch():
    return np.random.choice(np.arange(n_batches).astype(str), size=n_samples)


@pytest.fixture(scope="module")
def source_anndata(source_data, source_batch):
    return AnnData(source_data, obs=pd.DataFrame(source_batch, columns=["batch"]))


@pytest.fixture(scope="module")
def target_anndata(target_data, target_batch):
    return AnnData(target_data, obs=pd.DataFrame(target_batch, columns=["batch"]))


@pytest.fixture(scope="module")
def source_scvi_params():
    return {
        "model": {
            "dispersion": "gene-cell",
            "gene_likelihood": "zinb",
            "n_hidden": 10,
            "n_latent": n_latent,
            "n_layers": 1,
            "dropout_rate": 0.1,
        },
        "plan": {
            "lr": 0.005,
            "weight_decay": 0.01,
            "reduce_lr_on_plateau": True,
        },
        "train": {"early_stopping": True, "max_epochs": 10},
    }


@pytest.fixture(scope="module")
def target_scvi_params():
    return {
        "model": {
            "dispersion": "gene-cell",
            "gene_likelihood": "zinb",
            "n_hidden": 10,
            "n_latent": n_latent,
            "n_layers": 1,
            "dropout_rate": 0.1,
        },
        "plan": {
            "lr": 0.005,
            "weight_decay": 0.01,
            "reduce_lr_on_plateau": True,
        },
        "train": {"early_stopping": True, "max_epochs": 10},
    }


class TestSobolevAlignment:
    @pytest.fixture(scope="class")
    def sobolev_alignment_raw(self, falkon_import, source_scvi_params, target_scvi_params):
        if falkon_import:
            source_krr_params = {"method": "falkon"}
            target_krr_params = {"method": "falkon"}
        else:
            source_krr_params = {"method": "sklearn"}
            target_krr_params = {"method": "sklearn"}

        return SobolevAlignment(
            source_scvi_params=source_scvi_params,
            target_scvi_params=target_scvi_params,
            source_krr_params=source_krr_params,
            target_krr_params=target_krr_params,
            source_batch_name=None,
            target_batch_name=None,
            no_posterior_collapse=False,
        )

    @pytest.fixture(scope="class")
    def sobolev_alignment_batch(self, falkon_import, source_scvi_params, target_scvi_params):
        if falkon_import:
            source_krr_params = {"method": "falkon"}
            target_krr_params = {"method": "falkon"}
        else:
            source_krr_params = {"method": "sklearn"}
            target_krr_params = {"method": "sklearn"}

        return SobolevAlignment(
            source_scvi_params=source_scvi_params,
            target_scvi_params=target_scvi_params,
            source_krr_params=source_krr_params,
            target_krr_params=target_krr_params,
            source_batch_name="batch",
            target_batch_name="batch",
            n_artificial_samples=n_artificial_samples,
            frac_save_artificial=frac_save_artificial,
            lib_size_norm=True,
            no_posterior_collapse=False,
        )

    @pytest.fixture(scope="class")
    def scvi_raw_trained(self, source_anndata, target_anndata, sobolev_alignment_raw):
        return sobolev_alignment_raw.fit(
            X_source=source_anndata,
            X_target=target_anndata,
        )

    @pytest.fixture(scope="class")
    def scvi_batch_trained(self, source_anndata, target_anndata, sobolev_alignment_batch):
        return sobolev_alignment_batch.fit(X_source=source_anndata, X_target=target_anndata)

    @pytest.fixture(scope="class")
    def scvi_batch_trained_lib_size(self, source_anndata, target_anndata, sobolev_alignment_batch):
        return sobolev_alignment_batch.fit(
            X_source=source_anndata,
            X_target=target_anndata,
        )

    ###
    # TEST INIT METHODS
    ###

    def test_training_scvi_batch_trained(
        self,
        scvi_batch_trained,
    ):
        assert type(scvi_batch_trained.scvi_models) == dict
        for _, model in scvi_batch_trained.scvi_models.items():
            assert model.history["train_loss_epoch"].values[-1, 0] < model.history["train_loss_epoch"].values[0, 0]

        for x in scvi_batch_trained.artificial_samples_:
            assert scvi_batch_trained.artificial_samples_[x].shape[0] == n_artificial_samples * frac_save_artificial
            assert scvi_batch_trained.artificial_samples_[x].shape[1] == n_genes

        for x in scvi_batch_trained.artificial_embeddings_:
            assert scvi_batch_trained.artificial_embeddings_[x].shape[0] == n_artificial_samples * frac_save_artificial
            assert scvi_batch_trained.artificial_embeddings_[x].shape[1] == n_latent

    def test_training_scvi_batch_trained_lib_size(
        self,
        scvi_batch_trained_lib_size,
    ):
        assert type(scvi_batch_trained_lib_size.scvi_models) == dict
        for _, model in scvi_batch_trained_lib_size.scvi_models.items():
            assert model.history["train_loss_epoch"].values[-1, 0] < model.history["train_loss_epoch"].values[0, 0]

        for x in scvi_batch_trained_lib_size.artificial_samples_:
            assert (
                scvi_batch_trained_lib_size.artificial_samples_[x].shape[0]
                == n_artificial_samples * frac_save_artificial
            )
            assert scvi_batch_trained_lib_size.artificial_samples_[x].shape[1] == n_genes

        for x in scvi_batch_trained_lib_size.artificial_embeddings_:
            assert (
                scvi_batch_trained_lib_size.artificial_embeddings_[x].shape[0]
                == n_artificial_samples * frac_save_artificial
            )
            assert scvi_batch_trained_lib_size.artificial_embeddings_[x].shape[1] == n_latent

        # np.savetxt(open('source.csv', 'w'), scvi_batch_trained_lib_size.artificial_samples_['source'])
        # np.savetxt(open('target.csv', 'w'), scvi_batch_trained_lib_size.artificial_samples_['target'])
        # assert np.mean(np.sum(scvi_batch_trained_lib_size.artificial_samples_["source"], axis=1)) == np.mean(
        #     np.sum(scvi_batch_trained_lib_size.artificial_samples_["target"], axis=1)
        # )

    def test_KRR_scvi_trained(self, scvi_batch_trained):
        for x in scvi_batch_trained.artificial_samples_:
            assert scvi_batch_trained.approximate_krr_regressions_[x].sample_weights_.shape[1] == n_latent

    # def test_training_scvi_raw_trained(self, scvi_raw_trained):
    #     assert type(scvi_raw_trained.scvi_models) == dict
    #     for x, model in scvi_raw_trained.scvi_models.items():
    #         assert model.history['train_loss_epoch'].values[-1,0] < model.history['train_loss_epoch'].values[0,0]
