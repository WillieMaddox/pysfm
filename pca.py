import numpy as np


def compute(X, ndims):
    X = np.asarray(X)
    u, s, v = np.linalg.svd(X, full_matrices=False)
    return v[:ndims]


def project(X, subspace):
    return np.dot(X, subspace.T)
