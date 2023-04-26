import numpy as np
import pytest
import scipy
import torch
from sobolev_alignment import KRRApprox
from sobolev_alignment.krr_approx import FALKON_IMPORTED

n_samples_valid = 50
n_genes = 100
n_latent = 7
n_samples = 2000
penalization = 0.0001
pearson_threshold = 0.99
M = 500


@pytest.fixture(scope="module")
def falkon_import():
    return FALKON_IMPORTED


@pytest.fixture(scope="module")
def input():
    return torch.normal(0, 1, size=(n_samples, n_genes))


@pytest.fixture(scope="module")
def valid_input():
    return torch.normal(0, 1, size=(n_samples_valid, n_genes))


@pytest.fixture(scope="module")
def embedding_proj():
    return torch.normal(0, 1, size=(n_genes, n_latent))


@pytest.fixture(scope="module")
def embedding(input, embedding_proj):
    return input.matmul(embedding_proj)


@pytest.fixture(scope="module")
def valid_embedding(valid_input, embedding_proj):
    return valid_input.matmul(embedding_proj)


class TestKRRApprox:
    ###
    # SKLEARN CLFS
    ###

    @pytest.fixture(scope="class")
    def sklearn_rbf_KRR(self):
        return KRRApprox(
            method="sklearn", kernel="rbf", kernel_params={"gamma": 1 / (2 * n_genes)}, penalization=penalization
        )

    @pytest.fixture(scope="class")
    def sklearn_matern_KRR(self):
        return KRRApprox(
            method="sklearn",
            kernel="matern",
            kernel_params={"length_scale": n_genes, "nu": 1.5},
            penalization=penalization,
        )

    @pytest.fixture(scope="class")
    def sklearn_laplacian_KRR(self):
        return KRRApprox(
            method="sklearn", kernel="laplacian", kernel_params={"gamma": 1 / (2 * n_genes)}, penalization=penalization
        )

    @pytest.fixture(scope="class")
    def fit_sklearn_rbf_ridge(self, sklearn_rbf_KRR, input, embedding):
        return sklearn_rbf_KRR.fit(input, embedding)

    @pytest.fixture(scope="class")
    def fit_sklearn_laplacian_ridge(self, sklearn_laplacian_KRR, input, embedding):
        return sklearn_laplacian_KRR.fit(input, embedding)

    @pytest.fixture(scope="class")
    def fit_sklearn_matern_ridge(self, sklearn_matern_KRR, input, embedding):
        return sklearn_matern_KRR.fit(input, embedding)

    ###
    # FALKON CLFS
    ###

    @pytest.fixture(scope="class")
    def falkon_rbf_KRR(self, falkon_import):
        if falkon_import:
            return KRRApprox(
                method="falkon",
                kernel="rbf",
                kernel_params={"sigma": np.sqrt(2 * n_genes)},
                penalization=penalization,
                M=M,
            )
        else:
            return None

    @pytest.fixture(scope="class")
    def falkon_matern_KRR(self, falkon_import):
        if falkon_import:
            return KRRApprox(
                method="falkon",
                kernel="matern",
                kernel_params={"sigma": np.sqrt(2 * n_genes), "nu": 1.5},
                penalization=penalization,
                M=M,
            )
        else:
            return None

    @pytest.fixture(scope="class")
    def falkon_laplacian_KRR(self, falkon_import):
        if falkon_import:
            return KRRApprox(
                method="falkon",
                kernel="laplacian",
                kernel_params={"sigma": np.sqrt(2 * n_genes)},
                penalization=penalization,
                M=M,
            )
        else:
            return None

    @pytest.fixture(scope="class")
    def fit_falkon_rbf_ridge(self, falkon_rbf_KRR, input, embedding):
        if falkon_rbf_KRR is None:
            return None
        return falkon_rbf_KRR.fit(input, embedding)

    @pytest.fixture(scope="class")
    def fit_falkon_laplacian_ridge(self, falkon_laplacian_KRR, input, embedding):
        if falkon_laplacian_KRR is None:
            return None
        return falkon_laplacian_KRR.fit(input, embedding)

    @pytest.fixture(scope="class")
    def fit_falkon_matern_ridge(self, falkon_matern_KRR, input, embedding):
        if falkon_matern_KRR is None:
            return None
        return falkon_matern_KRR.fit(input, embedding)

    ###
    # TEST INIT METHODS
    ###

    def test_create_null_instance(self):
        KRRApprox()
        return True

    def test_all_sklearn_kernels(self):
        for kernel in KRRApprox.sklearn_kernel:
            KRRApprox(kernel=kernel, method="sklearn")
        return True

    def test_all_falkon_kernels(self, falkon_import):
        if falkon_import:
            print(">>>>>>\n\n\n\n")
            print(falkon_import)
            print("\n\n\n<<<<<")
            for kernel in KRRApprox.falkon_kernel:
                KRRApprox(kernel=kernel, method="falkon")
        return True

    ###
    # TEST FIT FOR SKLEARN
    ###

    def test_rbf_sklearn_fit(self, fit_sklearn_rbf_ridge, valid_input, valid_embedding):
        pred = fit_sklearn_rbf_ridge.transform(valid_input)
        pearson_corr = scipy.stats.pearsonr(pred.flatten(), valid_embedding.detach().numpy().flatten())
        assert pearson_corr[0] > pearson_threshold

    def test_matern_sklearn_fit(self, fit_sklearn_matern_ridge, valid_input, valid_embedding):
        pred = fit_sklearn_matern_ridge.transform(valid_input)
        pearson_corr = scipy.stats.pearsonr(pred.flatten(), valid_embedding.detach().numpy().flatten())
        assert pearson_corr[0] > pearson_threshold

    def test_laplacian_sklearn_fit(self, fit_sklearn_laplacian_ridge, valid_input, valid_embedding):
        pred = fit_sklearn_laplacian_ridge.transform(valid_input)
        pearson_corr = scipy.stats.pearsonr(pred.flatten(), valid_embedding.detach().numpy().flatten())
        assert pearson_corr[0] > pearson_threshold

    ###
    # TEST FIT FOR FALKON
    ###

    def test_rbf_falkon_fit(self, fit_falkon_rbf_ridge, valid_input, valid_embedding):
        if fit_falkon_rbf_ridge is not None:
            pred = fit_falkon_rbf_ridge.transform(valid_input)
            pearson_corr = scipy.stats.pearsonr(pred.flatten(), valid_embedding.detach().numpy().flatten())
            assert pearson_corr[0] > pearson_threshold

    def test_matern_falkon_fit(self, fit_falkon_matern_ridge, valid_input, valid_embedding):
        if fit_falkon_matern_ridge is not None:
            pred = fit_falkon_matern_ridge.transform(valid_input)
            pearson_corr = scipy.stats.pearsonr(pred.flatten(), valid_embedding.detach().numpy().flatten())
            assert pearson_corr[0] > pearson_threshold

    def test_laplacian_falkon_fit(self, fit_falkon_laplacian_ridge, valid_input, valid_embedding):
        if fit_falkon_laplacian_ridge is not None:
            pred = fit_falkon_laplacian_ridge.transform(valid_input)
            pearson_corr = scipy.stats.pearsonr(pred.flatten(), valid_embedding.detach().numpy().flatten())
            assert pearson_corr[0] > pearson_threshold

    def test_ridge_coef_sklearn_fit(self, fit_sklearn_laplacian_ridge, input, valid_input):
        if fit_sklearn_laplacian_ridge is not None:
            pred_reconstruct = fit_sklearn_laplacian_ridge.kernel_(
                valid_input, input[fit_sklearn_laplacian_ridge.ridge_samples_idx_, :]
            )
            pred_reconstruct = pred_reconstruct.dot(fit_sklearn_laplacian_ridge.sample_weights_)
            np.testing.assert_array_almost_equal(
                pred_reconstruct, fit_sklearn_laplacian_ridge.transform(valid_input), decimal=3
            )

    def test_ridge_coef_falkon_fit(self, fit_falkon_laplacian_ridge, input, valid_input):
        if fit_falkon_laplacian_ridge is not None:
            pred_reconstruct = fit_falkon_laplacian_ridge.kernel_(valid_input, fit_falkon_laplacian_ridge.anchors())
            pred_reconstruct = pred_reconstruct.matmul(fit_falkon_laplacian_ridge.sample_weights_)
            np.testing.assert_array_almost_equal(
                pred_reconstruct, fit_falkon_laplacian_ridge.transform(valid_input), decimal=3
            )
