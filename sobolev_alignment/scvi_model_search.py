"""
scVI model search.

@author: Soufiane Mourragui

Pipeline to perform model selection for the scVI model.
"""

import numpy as np
import pandas as pd
import scvi
from anndata import AnnData
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from sklearn.model_selection import train_test_split

from ._scvi_default_params import SCVI_MODEL_PARAMS, SCVI_PLAN_PARAMS, SCVI_TRAIN_PARAMS

DEFAULT_HYPEROPT_SPACE = {
    "n_hidden": hp.choice("n_hidden", [32, 64, 128, 256, 512, 1024]),
    "n_latent": 5 + hp.randint("n_latent", 20),
    "n_layers": 1 + hp.randint("n_layers", 5),
    "dropout_rate": hp.choice("dropout_rate", [0.0, 0.1, 0.3, 0.5, 0.7]),
    "gene_likelihood": hp.choice("gene_likelihood", ["zinb", "nb"]),
    "lr": hp.choice("learning_rate", [0.01, 0.005, 0.001, 0.0005, 0.0001]),
    "reduce_lr_on_plateau": hp.choice("reduce_lr_on_plateau", [True, False]),
    "early_stopping": hp.choice("early_stopping", [True, False]),
    "weight_decay": hp.choice("weight_decay", [0.01, 0.001, 0.0001, 0.00001, 0.0]),
    "dispersion": hp.choice("dispersion", ["gene", "gene-batch", "gene-cell"]),
}


def model_selection(
    data_an: AnnData,
    batch_key: str = None,
    model=scvi.model.SCVI,
    space=DEFAULT_HYPEROPT_SPACE,
    max_eval=100,
    test_size=0.1,
    save=None,
):
    """
    Model selection for scVI instances (hyper-parameter search).

    Perform model selection on an scVI model by dividing a dataset into training and
    testing, and subsequently performing Bayesian Optimisation on the test data.

    Parameters
    ----------
    data_an: AnnData
        Datasets to be used in the model selection.

    batch_key: str, default to None
        Name of the batch key to be used in scVI.

    model: default to scvi.model.SCVI
        Model from scvi-tools to be used.

    space: dict, default to DEFAULT_HYPEROPT_SPACE
        Dictionary with hyper-parameter space to be used in Bayesian optimisation.

    max_eval: int, default to 100
        Number of iterations in the Bayesian optimisation procedures, i.e., number
        of models assessed.

    test_size: float, default to 0.1
        Proportion of samples (cells) to be taken inside the test data.

    save: str, default to None
        Path to save Bayesian optimisation results to. Must be a csv file. If set
        to None, then results are not saved.

    Returns
    -------
        Tuple containing:
        - Best model given by hyperopt.
        - DataFrame with Bayesian optimisation results.
        - Trials instance from hyperopt.
    """
    # Remove batch if no batch_key (errors down the pipe otherwise)
    if batch_key is None and space == DEFAULT_HYPEROPT_SPACE:
        space["dispersion"] = hp.choice("dispersion", ["gene", "gene-cell"])

    # Split data randomly between train and test sets.
    train_data_an, test_data_an = split_dataset(data_an)

    # Bayesian optimisation
    save_hyperopt_res = Trials()
    _objective_function = make_objective_function(
        train_data_an=train_data_an, test_data_an=test_data_an, batch_key=batch_key, model=model
    )
    best = fmin(
        _objective_function, space, algo=tpe.suggest, max_evals=max_eval, return_argmin=False, trials=save_hyperopt_res
    )

    # Save
    hyperopt_results_df = pd.DataFrame(save_hyperopt_res.results)
    if save is not None:
        hyperopt_results_df.to_csv(save)

    return best, hyperopt_results_df, save_hyperopt_res


def make_objective_function(train_data_an, test_data_an, batch_key=None, model=scvi.model.SCVI):
    """
    Generate Hyperopt objective function.

    Generate the hyperopt objective function performing, for one set of hyperparameters,
    the training, the evaluation on test data and summing up all the results in a
    dictionary usable for Hyperopt.

    Parameters
    ----------
    train_data_an: AnnData
        AnnData containing the train samples.

    test_data_an: AnnData
         AnnData containing the test samples.

    batch_key: str, default to None
        Name of the batch key to be used in scVI.

    model: default to scvi.model.SCVI
        Model from scvi-tools to be used.

    Returns
    -------
        function which can be called using a dictionary of parameters.
    """
    train_data_an = train_data_an.copy()
    model.setup_anndata(train_data_an, batch_key=batch_key)

    def _objective_function(params):
        """
        Objective function.

        Returns a method which performs, for one set of hyperparameters, the training,
        the evaluation on test data and summing up all the results in a dictionary usable
        for Hyperopt.

        Parameters
        ----------
        params
            Dictionary with scvi params.

        Returns
        -------
            pd.DataFrame recapitulating the parameters (params) alongside the train and
            test loss.
        """
        # Structure parameters for scVI
        model_kwargs = {k: params[k] for k in params if k in SCVI_MODEL_PARAMS}
        plan_kwargs = {k: params[k] for k in params if k in SCVI_PLAN_PARAMS}
        trainer_kwargs = {k: params[k] for k in params if k in SCVI_TRAIN_PARAMS}

        # Initialize scVI model
        clf = model(train_data_an, **model_kwargs)

        try:
            clf.train(plan_kwargs=plan_kwargs, **trainer_kwargs)

            # Reconstruction error
            test_reconstruction_error = clf.get_reconstruction_error(adata=test_data_an)["reconstruction_loss"]
            train_reconstruction_error = clf.get_reconstruction_error()["reconstruction_loss"]

            # ELBO
            test_elbo = clf.get_elbo(adata=test_data_an)
            train_elbo = clf.get_elbo()
            del clf

            results_dict = {
                "loss": -test_elbo,
                "loss_choice": "ELBO",
                "train_reconstruction": train_reconstruction_error,
                "test_reconstruction": test_reconstruction_error,
                "train_ELBO": train_elbo.cpu().detach().numpy(),
                "test_ELBO": test_elbo.cpu().detach().numpy(),
                "status": STATUS_OK,
            }
        except ValueError:
            results_dict = {"status": STATUS_FAIL, "loss": np.iinfo(np.uint64).max}
            results_dict.update(params)
        except MisconfigurationException:
            # Observed here with incompatibility with ReduceLROnPlateau
            results_dict = {"status": STATUS_FAIL, "loss": np.iinfo(np.uint64).max}
            results_dict.update(params)

        results_dict.update(params)
        return results_dict

    return _objective_function


def split_dataset(data_an, test_size=0.1):
    """Split between training and testing."""
    train_data_df, test_data_df = train_test_split(data_an.to_df(), test_size=test_size)
    train_data_an = data_an[train_data_df.index,]
    test_data_an = data_an[test_data_df.index,]
    return train_data_an, test_data_an
