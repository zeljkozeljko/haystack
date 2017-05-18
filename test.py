import os
import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import islice
from reaper_algos import *
from reaper_translated import reaper_matlab
from plotting import *
from point_generator import *
from utils import extract_d_dim_principal_space

# Ambient dim
D = 300
# Number of outliers and inliers
N_outliers = 100
N_inliers = 15
d = 1
dim = 1

inliers, outliers, both_points, Pi_L = uniform_ball_subspace(D, N_outliers, dim, N_inliers,
                                                      s_out = 0.5, s_in = 0.15,
                                                      method = 'canonical')

# import pdb; pdb.set_trace()
delta = 1e-10
epsilon = 1e-15

P = reaper_matlab(both_points, d, spherical = "False")
PS = reaper_matlab(both_points, d, spherical = "True")

P_postprocessed = extract_d_dim_principal_space(P, dim)
PS_postprocessed = extract_d_dim_principal_space(PS, dim)

# # Errors
# errors = np.zeros(N_inliers + N_outliers)
# for i in range(len(errors)):
#     errors[i] = np.linalg.norm(both_points[i, :] - np.dot(P, both_points[i, :]))
#
# errors = np.zeros(N_inliers + N_outliers)
# for i in range(len(errors)):
#     errors[i] = np.linalg.norm(both_points[i, :] - np.dot(P_postprocessed, both_points[i, :]))
print "(Normal Reaper) Spectral norm = ", np.linalg.norm(Pi_L - P_postprocessed, 2)
print "(Spherical Reaper) Spectral norm = ", np.linalg.norm(Pi_L - PS_postprocessed, 2)

plot_points_and_projected(P_postprocessed, PS_postprocessed, Pi_L, inliers, outliers)


import pdb; pdb.set_trace()

# A = np.random.randn(N_outliers, D)
# weights = np.ones(N_outliers)
# P = weighted_least_squares(A, weights, d)
# Rank i.e. dimension of the linear space
