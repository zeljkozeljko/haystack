import os
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import time
from itertools import islice
from reaper_algos import *
from plotting import *
from point_generator import *
# Ambient dim
D = 100
# Number of outliers and inliers
N_outliers = 200
N_inliers = 10
d = 1

Inliers, Outliers, All = needle(D, N_outliers, N_inliers, s_out = 1, s_in = 0.25)
# import pdb; pdb.set_trace()
plot_separate(Inliers, Outliers)
import pdb; pdb.set_trace()
delta = 1e-10
epsilon = 1e-15

P = IRLS(All, d, delta, epsilon, spherical = "True")
errors = np.zeros(N_inliers + N_outliers)
for i in range(len(errors)):
    errors[i] = np.linalg.norm(All[i, :] - np.dot(P, All[i, :]))

Pi_L = np.zeros((D, D))
for i in range(d):
    Pi_L[i, i] = 1.0
print "Spectral norm is ", np.linalg.norm(Pi_L - P, 2)

import pdb; pdb.set_trace()

# A = np.random.randn(N_outliers, D)
# weights = np.ones(N_outliers)
# P = weighted_least_squares(A, weights, d)
# Rank i.e. dimension of the linear space
