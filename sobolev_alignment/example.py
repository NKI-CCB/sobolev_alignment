"""
<h2>Toy example for Sobolev Alignment</h2>

@author: Soufiane Mourragui
"""

import numpy as np
import pandas as pd
from anndata import AnnData

from sobolev_alignment import SobolevAlignment

# Generate data
n_source = 100
n_target = 200
n_features = 500

X_source = np.random.normal(size=(n_source, n_features))
X_source = np.exp(X_source + np.random.randint(3, 10, n_features)).astype(int)
X_source = AnnData(X_source, obs=pd.DataFrame(np.random.choice(["A", "B"], n_source).astype(str), columns=["pool"]))

X_target = np.random.normal(size=(n_target, n_features))
X_target = np.exp(X_target + np.random.randint(3, 10, n_features)).astype(int)
X_target = AnnData(X_target, obs=pd.DataFrame(np.random.choice(["A", "B"], n_target).astype(str), columns=["pool"]))

# Create a Sobolev Alignemnt instance
sobolev_alignment_clf = SobolevAlignment(
    source_scvi_params={"train": {"early_stopping": True}, "model": {}, "plan": {}},
    target_scvi_params={"train": {"early_stopping": True}, "model": {}, "plan": {}},
    n_jobs=2,
)

# Compute consensus features
sobolev_alignment_clf.fit(X_source, X_target, source_batch_name="pool", target_batch_name="pool")
