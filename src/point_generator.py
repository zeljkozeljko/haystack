import os
import time
from itertools import islice

import numpy as np
from scipy.stats import ortho_group

from reaper_algos import sampling_ratios


def gaussian_subspace(D, N_out, d, N_in, s_out=1.0, s_in=1.0,
                      s_noise_in = 0.0, pi_L = None, method='canonical'):
    """ D is the ambient dimension, N_in the number of inliers and  N_out the outliers
        d is the dimension of the desired subspace """

    print "Generating points for a " + str(d) + "-dimensional subpsace in a " + str(D) + "-dimensional ambient space"
    sampling_ratios(N_in, d, N_out, D)
    # Get orthoprojector, basis for L and basis for L_perp
    if pi_L is None:
        orthonormal_vectors, basis, pi_L = get_orthonormal_system_and_projector(D, d,
                                                                method = method)
    else:
        orthonormal_vectors, S, V = np.linalg.svd(pi_L)
    # Sample inliers
    factor_in = (s_in ** 2) / float(d)
    inliers = np.random.multivariate_normal(np.zeros(D), factor_in * pi_L, N_in)
    if s_noise_in > 0.0:
        factor_inlier_noise = s_noise_in * 1.0/(D - d)
        inlier_noise = np.random.multivariate_normal(np.zeros(D),
                                            factor_inlier_noise * (np.eye(D) - pi_L), N_in)
        inliers += inlier_noise
    # Sample outliers
    factor_out = (s_out ** 2) / float(D)
    outliers = np.random.multivariate_normal(np.zeros(D), factor_out * np.eye(D),
                                             N_out)
    return inliers, outliers, np.vstack((inliers, outliers)), pi_L


def uniform_subspace(D, N_out, d, N_in, s_out=1.0, s_in=1.0, pi_L = None,
                     s_noise_in = 0.0, method='canonical'):
    """ D is the ambient dimension, N_in the number of inliers and  N_out the outliers
        d is the dimension of the desired subspace """

    print "Generating points for a " + str(d) + "-dimensional subpsace in a " + str(D) + "-dimensional ambient space"
    sampling_ratios(N_in, d, N_out, D)
    # Get orthoprojector, basis for L and basis for L_perp
    if pi_L is None:
        orthonormal_vectors, basis, pi_L = get_orthonormal_system_and_projector(D, d,
                                                                method = method)
    else:
        orthonormal_vectors, S, V = np.linalg.svd(pi_L)
    # Sample inliers
    inliers = (pi_L.dot(np.random.uniform(low=-s_in, high=s_in, size=(D, N_in)))).T
    if s_noise_in > 0.0:
        factor_inlier_noise = s_noise_in * 1.0/(D - d)
        inlier_noise = np.random.multivariate_normal(np.zeros(D),
                                            factor_inlier_noise * (np.eye(D) - pi_L), N_in)
        inliers += inlier_noise
    # Sample outliers
    outliers = (pi_L.dot(np.random.uniform(low=-s_in, high=s_in, size=(D, N_out)))).T
    noise_coeffs_coefs = np.random.uniform(low=-s_out, high=s_out, size=(N_out, D - d))
    for j in range(N_out):
        outliers[j, :] += orthonormal_vectors[:, d:].dot(noise_coeffs_coefs[j, :])
    return inliers, outliers, np.vstack((inliers, outliers)), pi_L


def uniform_ball_subspace(D, N_out, d, N_in, s_out=1.0, s_in=1.0, pi_L = None,
                          s_noise_in = 0.0, method='canonical'):
    """ D is the ambient dimension, N_in the number of inliers and  N_out the outliers
        d is the dimension of the desired subspace """

    print "Generating points for a " + str(d) + "-dimensional subpsace in a " + str(D) + "-dimensional ambient space"
    sampling_ratios(N_in, d, N_out, D)
    # Get orthoprojector, basis for L and basis for L_perp
    if pi_L is None:
        orthonormal_vectors, basis, pi_L = get_orthonormal_system_and_projector(D, d,
                                                                method = method)
    else:
        orthonormal_vectors, S, V = np.linalg.svd(pi_L)
    # Sample inliers
    inliers = (pi_L.dot(np.random.uniform(low=-s_in, high=s_in, size=(D, N_in)))).T
    if s_noise_in > 0.0:
        factor_inlier_noise = s_noise_in * 1.0/(D - d)
        inlier_noise = np.random.multivariate_normal(np.zeros(D),
                                            factor_inlier_noise * (np.eye(D) - pi_L), N_in)
        inliers += inlier_noise
    # Sample outliers
    outliers = (pi_L.dot(np.random.uniform(low=-s_in, high=s_in, size=(D, N_out)))).T
    # Noise in normal space: Get coefficients
    coefs = np.random.normal(size=(D - d, N_out))
    for j in range(N_out):
        coefs[:, j] = coefs[:, j] / np.linalg.norm(coefs[:, j]) * s_out
        outliers[j, :] += orthonormal_vectors[:, d:].dot(coefs[:, j])
    return inliers, outliers, np.vstack((inliers, outliers)), pi_L

def get_orthonormal_system_and_projector(ambient_dim, intrinsic_dim,
                                         method = 'canonical'):
    if method == 'canonical':
        orthonormal_vectors = np.eye(ambient_dim)
    elif method == 'random':
        orthonormal_vectors = ortho_group.rvs(ambient_dim)
    basis_L = orthonormal_vectors[:, 0:intrinsic_dim]
    pi_L = basis_L.dot(basis_L.T)
    return orthonormal_vectors, basis_L, pi_L
