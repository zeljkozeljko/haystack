import numpy as np

from point_generator import (gaussian_subspace, uniform_ball_subspace,
                             uniform_subspace)
from reaper_translated import reaper_matlab
from utils import extract_d_dim_principal_space
from preprocessing import energy_filter

def run_error_analysis(ambient_dim, intrinsic_dim, inlier_points,
                       outlier_points, outlier_spreads, inlier_spreads,
                       repititions, sampling, method,
                       preprocessing = None,
                       cutoff = 0.0):
    n_inlier_points = len(inlier_points)
    n_outlier_points = len(outlier_points)
    n_inlier_spreads = len(inlier_spreads)
    n_outlier_spreads = len(outlier_spreads)
    errors_non_spherical = np.zeros((n_inlier_points, n_outlier_points,
                                     n_inlier_spreads, n_outlier_spreads,
                                     repititions))
    errors_spherical = np.zeros((n_inlier_points, n_outlier_points,
                                     n_inlier_spreads, n_outlier_spreads,
                                     repititions))
    total_n_exp = n_inlier_points * n_outlier_points * n_inlier_spreads * \
                    n_outlier_spreads * repititions
    ctr = 0
    # Get sampling
    if sampling == 'uniform':
        sampling_method = uniform_subspace
    elif sampling == 'uniform_ball':
        sampling_method = uniform_ball_subspace
    elif sampling == 'gaussian':
        sampling_method = gaussian_subspace
    for i, n_inlier in enumerate(inlier_points):
        for j, n_outlier in enumerate(outlier_points):
            for k, inlier_spread in enumerate(inlier_spreads):
                for l, outlier_spread in enumerate(outlier_spreads):
                    for rep in range(repititions):
                        print "Experiment {0}/{1}".format(ctr, total_n_exp)
                        inliers, outliers, both_points, Pi_L = sampling_method(
                                                            ambient_dim,
                                                            n_outlier,
                                                            intrinsic_dim,
                                                            n_inlier,
                                                            s_out = outlier_spread,
                                                            s_in = inlier_spread,
                                                            method = method)
                        if preprocessing == 'energy':
                            both_points = energy_filter(both_points, cutoff)
                        P = reaper_matlab(both_points, intrinsic_dim, spherical = "False")
                        PS = reaper_matlab(both_points, intrinsic_dim, spherical = "True")
                        P_postprocessed = extract_d_dim_principal_space(P, intrinsic_dim)
                        PS_postprocessed = extract_d_dim_principal_space(PS, intrinsic_dim)
                        errors_non_spherical[i,j,k,l,rep] = np.linalg.norm(Pi_L - P_postprocessed, 2)
                        errors_spherical[i,j,k,l,rep] = np.linalg.norm(Pi_L - PS_postprocessed, 2)
                        ctr += 1
                        print "Non-spherical Error: ", errors_non_spherical[i,j,k,l,rep]
                        print "Spherical Error: ", errors_spherical[i,j,k,l,rep]
    return errors_non_spherical, errors_spherical

def run_error_analysis_inlier_noise(ambient_dim, intrinsic_dim, inlier_points,
                                    outlier_points, outlier_spreads, inlier_spreads,
                                    inlier_noise_spreads,
                                    repititions, sampling, method,
                                    preprocessing = None,
                                    cutoff = 0.0):
    n_inlier_points = len(inlier_points)
    n_outlier_points = len(outlier_points)
    n_inlier_spreads = len(inlier_spreads)
    n_outlier_spreads = len(outlier_spreads)
    n_inlier_noise_spreads = len(inlier_noise_spreads)
    errors_non_spherical = np.zeros((n_inlier_points, n_outlier_points,
                                     n_inlier_spreads, n_outlier_spreads,
                                     n_inlier_noise_spreads, repititions))
    errors_spherical = np.zeros((n_inlier_points, n_outlier_points,
                                     n_inlier_spreads, n_outlier_spreads,
                                     n_inlier_noise_spreads, repititions))
    total_n_exp = n_inlier_points * n_outlier_points * n_inlier_spreads * \
                    n_outlier_spreads * n_inlier_noise_spreads * repititions
    ctr = 0
    # Get sampling
    if sampling == 'uniform':
        sampling_method = uniform_subspace
    elif sampling == 'uniform_ball':
        sampling_method = uniform_ball_subspace
    elif sampling == 'gaussian':
        sampling_method = gaussian_subspace
    for i, n_inlier in enumerate(inlier_points):
        for j, n_outlier in enumerate(outlier_points):
            for k, inlier_spread in enumerate(inlier_spreads):
                for l, outlier_spread in enumerate(outlier_spreads):
                    for m, inlier_noise_spread in enumerate(inlier_noise_spreads):
                        for rep in range(repititions):
                            print "Experiment {0}/{1}".format(ctr, total_n_exp)
                            inliers, outliers, both_points, Pi_L = sampling_method(
                                                                ambient_dim,
                                                                n_outlier,
                                                                intrinsic_dim,
                                                                n_inlier,
                                                                s_out = outlier_spread,
                                                                s_in = inlier_spread,
                                                                s_noise_in = inlier_noise_spread,
                                                                method = method)
                            if preprocessing == 'energy':
                                both_points = energy_filter(both_points, cutoff)
                            P = reaper_matlab(both_points, intrinsic_dim, spherical = "False")
                            PS = reaper_matlab(both_points, intrinsic_dim, spherical = "True")
                            P_postprocessed = extract_d_dim_principal_space(P, intrinsic_dim)
                            PS_postprocessed = extract_d_dim_principal_space(PS, intrinsic_dim)
                            errors_non_spherical[i,j,k,l,m,rep] = np.linalg.norm(Pi_L - P_postprocessed, 2)/np.linalg.norm(Pi_L, 2)
                            errors_spherical[i,j,k,l,m,rep] = np.linalg.norm(Pi_L - PS_postprocessed, 2)/np.linalg.norm(Pi_L, 2)
                            ctr += 1
                            print "Non-spherical Error: ", errors_non_spherical[i,j,k,l,m,rep]
                            print "Spherical Error: ", errors_spherical[i,j,k,l,m,rep]
    return errors_non_spherical, errors_spherical
