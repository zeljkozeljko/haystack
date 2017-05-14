import os
import numpy as np
from termcolor import colored
import time
from itertools import islice
from reaper_algos import *

def needle(D, N_out, N_in, s_out = 1.0, s_in = 1.0, Pi_L = "Not given"):
    """ D is the ambient dimension, N_in the number of inliers and  N_out the outliers
        d is 1 """

    d = 1
    print "Generating points for a needle in a " + str(D) + "-dimensional ambient space"
    sampling_ratios(N_in, d, N_out, D)
    # import pdb; pdb.set_trace()


    # inliers
    if Pi_L == "Not given":
        Pi_L = np.zeros((D, D))
        Pi_L[0, 0] = 1.0

    factor_in = (s_in ** 2) / float(d)
    Inliers  = np.random.multivariate_normal(np.zeros(D), factor_in * Pi_L, N_in)

    ## Outliers
    factor_out = (s_out ** 2) / float(D)
    Outliers = np.random.multivariate_normal(np.zeros(D), factor_out * np.eye(D), N_out)

    return Inliers, Outliers, np.vstack((Inliers, Outliers))

def sheet(D, N_out, N_in, s_out = 1.0, s_in = 1.0, Pi_L = "Not given"):
    """ D is the ambient dimension, N_in the number of inliers and  N_out the outliers
        d is 2 """

    d = 2
    print "Generating points for a sheet in a " + str(D) + "-dimensional ambient space"
    sampling_ratios(N_in, d, N_out, D)
    # import pdb; pdb.set_trace()


    # inliers
    if Pi_L == "Not given":
        Pi_L = np.zeros((D, D))
        Pi_L[0, 0] = 1.0
        Pi_L[1, 1] = 1.0

    factor_in = (s_in ** 2) / float(d)
    Inliers  = np.random.multivariate_normal(np.zeros(D), factor_in * Pi_L, N_in)

    ## Outliers
    factor_out = (s_out ** 2) / float(D)
    Outliers = np.random.multivariate_normal(np.zeros(D), factor_out * np.eye(D), N_out)

    return Inliers, Outliers, np.vstack((Inliers, Outliers))


def subspace(D, N_out, d, N_in, s_out = 1.0, s_in = 1.0, Pi_L = "Not given"):
    """ D is the ambient dimension, N_in the number of inliers and  N_out the outliers
        d is the dimension of the desired subspace """

    print "Generating points for a " + str(d) + "-dimensional subpsace in a " + str(D) + "-dimensional ambient space"
    sampling_ratios(N_in, d, N_out, D)
    # import pdb; pdb.set_trace()

    # inliers
    if Pi_L == "Not given":
        Pi_L = np.zeros((D, D))
        for i in range(d):
            Pi_L[i, i] = 1.0

    factor_in = (s_in ** 2) / float(d)
    Inliers  = np.random.multivariate_normal(np.zeros(D), factor_in * Pi_L, N_in)

    ## Outliers
    factor_out = (s_out ** 2) / float(D)
    Outliers = np.random.multivariate_normal(np.zeros(D), factor_out * np.eye(D), N_out)

    return Inliers, Outliers, np.vstack((Inliers, Outliers))
