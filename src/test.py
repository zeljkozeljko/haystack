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
from error_analysis import run_error_analysis

# Dimensions
ambient_dim = 200
intrinsic_dim = 1

# Number of outliers and inliers
outlier_points = [50]
inlier_points = [10]

# Spreads
outlier_spreads = np.linspace(1.0/200.0 * 0.5, 0.5, 20)
inlier_spreads = [0.5]

# repititions
repititions = 1

# Sampling
sampling = "uniform"
method = 'random'

# Preprocessing
preprocessing = 'energy'
cutoff = (0.5 * np.float(intrinsic_dim)) ** 2

# Plot against
y_index = 3
# Calculate Errors
error_normal, errors_spherical = run_error_analysis(ambient_dim,
                   intrinsic_dim, inlier_points, outlier_points,
                   outlier_spreads, inlier_spreads, repititions, sampling,
                   method, preprocessing = preprocessing, cutoff = cutoff)

errors_mean_normal = np.mean(error_normal, axis = 4)
errors_std_normal = np.std(error_normal, axis = 4)
errors_mean_spherical = np.mean(errors_spherical, axis = 4)
errors_std_spherical = np.std(errors_spherical, axis = 4)

plot_error(errors_mean_normal, errors_std_normal,
           errors_mean_spherical, errors_std_spherical,
           ambient_dim, intrinsic_dim, inlier_points, outlier_points,
           outlier_spreads, inlier_spreads, repititions, sampling, method,
           y_index, preprocessing, cutoff)
