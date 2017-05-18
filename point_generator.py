import os
import time
from itertools import islice

import numpy as np
from scipy.stats import ortho_group

from reaper_algos import sampling_ratios


def subspace(D, N_out, d, N_in, s_out = 1.0, s_in = 1.0, Pi_L = "Not given",
             method = 'canonical'):
    """ D is the ambient dimension, N_in the number of inliers and  N_out the outliers
        d is the dimension of the desired subspace """

    print "Generating points for a " + str(d) + "-dimensional subpsace in a " + str(D) + "-dimensional ambient space"
    sampling_ratios(N_in, d, N_out, D)
    # import pdb; pdb.set_trace()

    # inliers
    if Pi_L == "Not given":
        Pi_L = np.zeros((D, D))
        if method == 'canonical':
            for i in range(d):
                Pi_L[i, i] = 1.0
        elif method == 'random':
            orthonormal_vectors = ortho_group.rvs(D)
            basis = orthonormal_vectors[:,0:d]
            Pi_L = basis.dot(basis.T)


    factor_in = (s_in ** 2) / float(d)
    Inliers  = np.random.multivariate_normal(np.zeros(D), factor_in * Pi_L, N_in)

    ## Outliers
    factor_out = (s_out ** 2) / float(D)
    Outliers = np.random.multivariate_normal(np.zeros(D), factor_out * np.eye(D), N_out)

    return Inliers, Outliers, np.vstack((Inliers, Outliers)), Pi_L

def uniform_subspace(D, N_out, d, N_in, s_out = 1.0, s_in = 1.0, Pi_L = "Not given",
                     method = 'canonical'):
    """ D is the ambient dimension, N_in the number of inliers and  N_out the outliers
        d is the dimension of the desired subspace """

    print "Generating points for a " + str(d) + "-dimensional subpsace in a " + str(D) + "-dimensional ambient space"
    sampling_ratios(N_in, d, N_out, D)

    # inliers
    if Pi_L == "Not given":
        Pi_L = np.zeros((D, D))
        if method == 'canonical':
            for i in range(d):
                Pi_L[i, i] = 1.0
        elif method == 'random':
            orthonormal_vectors = ortho_group.rvs(D)
            basis = orthonormal_vectors[:,0:d]
            Pi_L = basis.dot(basis.T)

    Inliers = (Pi_L.dot(np.random.uniform(low = -s_in, high = s_in, size = (D, N_in)))).T

    ## Outliers
    Outliers = (Pi_L.dot(np.random.uniform(low = -s_in, high = s_in, size = (D, N_out)))).T
    noise = np.random.uniform(low = -s_out, high = s_out, size = (N_out, D))
    noise[:,0:d] = 0.0
    Outliers += noise
    return Inliers, Outliers, np.vstack((Inliers, Outliers)), Pi_L

def uniform_ball_subspace(D, N_out, d, N_in, s_out = 1.0, s_in = 1.0, Pi_L = "Not given",
                          method = 'canonical'):
    """ D is the ambient dimension, N_in the number of inliers and  N_out the outliers
        d is the dimension of the desired subspace """

    print "Generating points for a " + str(d) + "-dimensional subpsace in a " + str(D) + "-dimensional ambient space"
    sampling_ratios(N_in, d, N_out, D)

    # inliers
    if Pi_L == "Not given":
        Pi_L = np.zeros((D, D))
        if method == 'canonical':
            orthonormal_vectors = np.eye(D)
            basis = orthonormal_vectors[:, 0:d]
            Pi_L = basis.dot(basis.T)
        elif method == 'random':
            orthonormal_vectors = ortho_group.rvs(D)
            basis = orthonormal_vectors[:,0:d]
            Pi_L = basis.dot(basis.T)

    Inliers = (Pi_L.dot(np.random.uniform(low = -s_in, high = s_in, size = (D, N_in)))).T

    ## Outliers
    # Remark: Basis of normal space is given by orthonormal_vectors[:,d:]
    # Base points
    Outliers = (Pi_L.dot(np.random.uniform(low = -s_in, high = s_in, size = (D, N_out)))).T
    # Noise in normal space: Get coefficients
    coefs = np.random.normal(size=(D-d, N_out))
    for j in range(N_out):
        coefs[:,j] = coefs[:,j]/np.linalg.norm(coefs[:,j]) * s_out
        Outliers[j,:] += orthonormal_vectors[:, d:].dot(coefs[:,j])
    return Inliers, Outliers, np.vstack((Inliers, Outliers)), Pi_L
