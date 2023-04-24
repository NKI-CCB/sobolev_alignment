"""
Kernel operations.

@author: Soufiane Mourragui

Custom scripts for specific matrix operations.
"""
import numpy as np


def mat_inv_sqrt(M, threshold=1e-6):
    """Compute the inverse square root of a symmetric matrix M by SVD."""
    u, s, v = np.linalg.svd(M)
    s = [1.0 / np.sqrt(x) if x > threshold else 0 for x in s]
    return v.T.dot(np.diag(s)).dot(u.T)
