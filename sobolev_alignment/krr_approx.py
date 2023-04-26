"""
Encoder approximation by Kernel Ridge Regression.

@author: Soufiane Mourragui

This modules train a Kernel Ridge Regression (KRR) on a pair of samples
(x_hat) and embeddings (z_hat) using two possible implementations:
- scikit-learn: deterministic, but limited in memory and time efficiency.
- Falkon: stochastic Nyström approximation, faster both in memory
and computation time. Optimised for multi GPUs.

References
----------
Mourragui et al 2022
Meanti et al, Kernel methods through the roof: handling billions of points efficiently,
NeurIPS, 2020.
Pedregosa et al, Scikit-learn: Machine Learning in Python, Journal of Machine Learning
Research, 2011.
"""

import gc
import os
from pickle import dump, load

import numpy as np
import torch

# Falkon import if loaded
try:
    from falkon import Falkon
    from falkon.kernels import GaussianKernel, LaplacianKernel, MaternKernel
    from falkon.options import FalkonOptions

    FALKON_IMPORTED = True
except ImportError:
    FALKON_IMPORTED = False
    print("FALKON NOT INSTALLED, OR NOT IMPORTED. USING FALKON WOULD RESULT IN BETTER PERFORMANCE.", flush=True)
from sklearn.gaussian_process.kernels import Matern, PairwiseKernel
from sklearn.kernel_ridge import KernelRidge

# Scikit-learn import
from sklearn.preprocessing import StandardScaler


class KRRApprox:
    """
    Kernel Ridge Regression approximation.

    This class contains the functions used to approximate the encoding functions of a Variational
    Auto Encoder (VAE) by a kernel machines by means of Kernel Ridge Regression (KRR).
    <br/>
    This class takes as input a training data and executes the learning process. The generation of
    artificial samples and subsequent computation of embeddings is not part of this class.
    """

    sklearn_kernel = {
        "rbf": "wrapper",
        "gaussian": "wrapper",
        "laplacian": "wrapper",
        "matern": Matern,
    }

    falkon_kernel = {
        "rbf": GaussianKernel if FALKON_IMPORTED else None,
        "gaussian": GaussianKernel if FALKON_IMPORTED else None,
        "laplacian": LaplacianKernel if FALKON_IMPORTED else None,
        "matern": MaternKernel if FALKON_IMPORTED else None,
    }

    default_kernel_params = {
        "falkon": {
            "rbf": {"sigma": 1},
            "gaussian": {"sigma": 1},
            "laplacian": {"sigma": 1},
            "matern": {"sigma": 1, "nu": 0.5},
        },
        "sklearn": {"rbf": {}, "gaussian": {}, "laplacian": {}, "matern": {}},
    }

    def __init__(
        self,
        method: str = "sklearn",
        kernel: str = "rbf",
        M: int = 100,
        kernel_params: dict = None,
        penalization: float = 1e-6,
        maxiter: int = 20,
        falkon_options: dict = None,
        mean_center: bool = False,
        unit_std: bool = False,
    ):
        """
        Class to perform latent space approximation by Kernel Ridge Regression.

        Parameters
        ----------
        method: str, default to 'sklearn'.
            Method used for KRR approximation, 'sklearn' or 'falkon'.

        kernel: str, default to 'rbf'
            Name of the kernel to use in the approximation: 'rbf' (Gaussian kernel),
            'matern' (Matern kernel), 'laplace' (Laplace).

        M: int, default to 100
            Number of anchors samples for the Nyström approximation. Only used when
            method='falkon'. Default to 100 corresponds to a very low number of anchors
            and would lead to a poor approximation.

        kernel_params: dict, default to None
            Dictionary containing the kernel hyper-parameters, e.g. nu or sigma parameters
            for a Matérn kernel.

        penalization: float, default to 1e-6
            Amount of penalization. Higher values induce more penalization. When method='sklearn',
            it corresponds to alpha (sklearn.kernel_ridge.KernelRidge).

        maxiter: int, default to 20
            Maximum number of iterations in Falkon.

        falkon_options: dictionary, default to None
            Additional options to provide to falkon.options.FalkonOptions. More information on
            Falkon's documentation:
            https://falkonml.github.io/falkon/api_reference/options.html#falkonoptions

        mean_center: bool, default to False
            Whether to mean center features (genes) before regression.

        unit_std: bool, default to False
            Whether to standardize features (genes) before regression.

        """
        self.method = method

        # Set kernel
        self.kernel = kernel
        self.kernel_params = kernel_params if kernel_params else self.default_kernel_params[self.method][self.kernel]
        self._make_kernel()

        # Set penalization parameters
        self.penalization = penalization
        self.M = M
        self.maxiter = maxiter

        # Set hardware specifications
        self.falkon_options = falkon_options if falkon_options else {}

        # Preprocessing
        self.mean_center = mean_center
        self.unit_std = unit_std
        self.pre_process_ = StandardScaler(with_mean=mean_center, with_std=unit_std, copy=False)

    def _make_kernel(self):
        """
        Process the kernel parameters and set up the kernel to be used in KRR.

        Returns
        -------
        True if kernel has successfully been set up.
        """
        # scikit-learn initialization
        if self.method.lower() == "sklearn":
            if self.sklearn_kernel[self.kernel.lower()] != "wrapper":
                self.kernel_ = self.sklearn_kernel[self.kernel.lower()](**self.kernel_params)
            else:
                self.kernel_ = PairwiseKernel(metric=self.kernel.lower(), **self.kernel_params)

        # Falkon
        elif self.method.lower() == "falkon":
            self.kernel_ = self.falkon_kernel[self.kernel.lower()](**self.kernel_params)

        # If not implemented
        else:
            raise NotImplementedError("%s not implemented. Choices: sklearn and falkon" % (self.method))

        return True

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Train a regression model (KRR) between X and all columns of Y.

        Parameters
        ----------
        X: torch.Tensor
            Tensor containing the artificial input (x_hat), with samples in the rows.

        y: torch.Tensor
            Tensor containing the artificial embedding (z_hat). Called y for compliance with
            sklearn functions.

        Returns
        -------
        self: fitted KRRApprox instance.
        """
        self._setup_clf()

        # Process data if required. No direct use of StandardScaler when mean_center and unit_std
        # are False as it can have a large memory footprint.
        if self.mean_center or self.unit_std:
            self.pre_process_.fit(X)
            self.training_data_ = torch.Tensor(self.pre_process_.transform(torch.Tensor(X)))
        else:
            self.training_data_ = X

        self.training_label_ = y

        # Train KRR instances.
        if self.method == "sklearn":
            self.ridge_clf_.fit(self.kernel_(self.training_data_), y)
        elif self.method == "falkon":
            self.ridge_clf_.fit(self.training_data_, y)

        self._save_coefs()

        if self.method == "falkon":
            self.training_data_ = self.ridge_clf_.ny_points_
        gc.collect()

        return self

    def _setup_clf(self):
        if self.method.lower() == "sklearn":
            self._setup_sklearn_clf()
        elif self.method.lower() == "falkon":
            self._setup_falkon_clf()

    def _setup_sklearn_clf(self):
        """
        Set up the regression model using scikit-learn implementation.

        Returns
        -------
        True if set up ran properly.
        """
        self.ridge_clf_ = KernelRidge(kernel="precomputed", alpha=self.penalization)
        return True

    def _setup_falkon_clf(self):
        """
        Set up the regression model using Falkon implementation.

        Returns
        -------
        True if set up ran properly.
        """
        self.ridge_clf_ = Falkon(
            kernel=self.kernel_,
            penalty=self.penalization,
            M=self.M,
            maxiter=self.maxiter,
            options=FalkonOptions(**self.falkon_options) if FALKON_IMPORTED else None,
        )
        return True

    def _save_coefs(self):
        if self.method.lower() == "sklearn":
            self._process_coef_ridge_sklearn()
        elif self.method.lower() == "falkon":
            self._process_coef_ridge_falkon()

    def _process_coef_ridge_sklearn(self):
        """Save and transform to torch.Tensor KRR coefficients from scikit-learn implementation."""
        self.sample_weights_ = torch.Tensor(self.ridge_clf_.dual_coef_)
        self.ridge_samples_idx_ = np.arange(self.training_data_.shape[0])

    def _process_coef_ridge_falkon(self):
        """Save the coefficients obtained after kernel ridge regression with Falkon implementation."""
        self.sample_weights_ = self.ridge_clf_.alpha_

    def anchors(self):
        """Return anchor points used in KRR."""
        return self.training_data_

    def transform(self, X: torch.Tensor):
        """
        Apply the trained KRR models to a given data.

        This corresponds to the out-of-sample extension.

        Parameters
        ----------
        X: torch.Tensor
            Tensor containing gene expression profiles with samples in the rows.
            <b>WARNING:</b> genes (features) need to be following the same order
            as the training data.

        Returns
        -------
        torch.Tensor with predicted values for each of the encoding functions.
        Samples are in the rows and encoding functions (embedding) in the columns.
        """
        if self.mean_center or self.unit_std:
            X = torch.Tensor(self.pre_process_.transform(X))

        if self.method == "sklearn":
            return self.ridge_clf_.predict(self.kernel_(X, self.training_data_))
        elif self.method == "falkon":
            return self.ridge_clf_.predict(X)
        else:
            raise NotImplementedError("%s not implemented. Choices: sklearn and falkon" % (self.method))

    def save(self, folder: str = "."):
        """
        Save the instance.

        Parameters
        ----------
        folder: str, default to '.'
            Folder path to use for saving the instance

        Returns
        -------
        True if the instance was properly saved.
        """
        # Create path should it not exist yet.
        if not os.path.exists(folder) and not os.path.isdir(folder):
            os.mkdir(folder)

        # Save parameters of KRR.
        params = {
            "method": self.method,
            "kernel": self.kernel_,
            "M": self.M,
            "penalization": self.penalization,
            "mean_center": self.mean_center,
            "unit_std": self.unit_std,
        }
        params.update(self.kernel_params)
        dump(params, open("%s/params.pkl" % (folder), "wb"))

        # Save important material:
        #   - KRR weights
        #   - Samples used for prediction.
        torch.save(torch.Tensor(self.anchors()), open("%s/sample_anchors.pt" % (folder), "wb"))
        torch.save(torch.Tensor(self.sample_weights_), open("%s/sample_weights.pt" % (folder), "wb"))

        # Save weights and anchors as csv.
        # Longer to load, but compatible with all platforms.
        np.savetxt("%s/sample_weights.csv" % (folder), self.sample_weights_.detach().numpy())
        np.savetxt("%s/sample_anchors.csv" % (folder), self.anchors().detach().numpy())

        return True

    def load(folder: str = "."):
        """
        Load a KRRApprox instance.

        Parameters
        ----------
        folder: str, default to '.'
            Folder path where the instance is located

        Returns
        -------
        KRRApprox: instance saved at the folder location.
        """
        # Load and format parameters.
        params = load(open("%s/params.pkl" % (folder), "rb"))
        krr_params = {
            e: f for e, f in params.items() if e in ["method", "M", "penalization", "mean_center", "unit_std"]
        }
        # krr_params['kernel'] = krr_params['kernel'].kernel_name
        krr_approx_clf = KRRApprox(**krr_params)
        krr_approx_clf.kernel_ = params["kernel"]

        # Load sample weights and anchors.
        krr_approx_clf.sample_weights_ = torch.load(open("%s/sample_weights.pt" % (folder), "rb"))
        krr_approx_clf.training_data_ = torch.load(open("%s/sample_anchors.pt" % (folder), "rb"))

        # Set up classifiers for out-of-sample application.
        krr_approx_clf._setup_clf()
        krr_approx_clf.ridge_clf_.ny_points_ = krr_approx_clf.training_data_
        krr_approx_clf.ridge_clf_.alpha_ = krr_approx_clf.sample_weights_

        return krr_approx_clf
