"""
Compute interpolated features.

@author: Soufiane Mourragui
"""

import numpy as np
import pandas as pd
import scipy


def compute_optimal_tau(PV_number, pv_projections, principal_angles, n_interpolation=100):
    """Compute the optimal interpolation step for each PV (Grassmann interpolation)."""
    ks_statistics = {}
    for tau_step in np.linspace(0, 1, n_interpolation + 1):
        source_proj, target_proj = project_on_interpolate_PV(
            principal_angles[PV_number], PV_number, tau_step, pv_projections
        )
        ks_statistics[tau_step] = scipy.stats.ks_2samp(source_proj, target_proj)
    ks_statistics = pd.DataFrame(ks_statistics, index=["ks", "pval"]).T.reset_index()
    optimal_tau = ks_statistics.sort_values("ks").iloc[0]["index"]

    return optimal_tau


def project_on_interpolate_PV(angle, PV_number, tau_step, pv_projections):
    """Project data on interpolated PVs."""
    source_proj = np.sin((1 - tau_step) * angle) * pv_projections["source"]["source"][:, PV_number]
    source_proj += np.sin(tau_step * angle) * pv_projections["target"]["source"][:, PV_number]
    source_proj /= np.sin(angle)

    target_proj = np.sin((1 - tau_step) * angle) * pv_projections["source"]["target"][:, PV_number]
    target_proj += np.sin(tau_step * angle) * pv_projections["target"]["target"][:, PV_number]
    target_proj /= np.sin(angle)

    return source_proj, target_proj
