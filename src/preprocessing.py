# coding: utf8
import numpy as np

def energy_filter(X, cutoff):
    n, D = X.shape
    filtered_indices = np.where(np.square(np.linalg.norm(X, axis = 1)) <= cutoff)[0]
    return X[filtered_indices, :]
