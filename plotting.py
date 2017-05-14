import os
import numpy as np
from termcolor import colored
import time
from itertools import islice
import matplotlib.pyplot as plt
from reaper_algos import *

def plot_all(X):

    N = X.shape[0]
    D = X.shape[1]

    fig = plt.figure()
    ax = fig.gca()
    cmap = plt.cm.gist_ncar
    title = "All of the points in a " + str(D) + "-dimensional ambient space"
    sc = ax.scatter(X[:, 0], X[:, 3], cmap = cmap)
    plt.title(title)
    plt.show()

def plot_separate(X_in, X_out):

    N_in = X_in.shape[0]
    N_out = X_out.shape[0]
    D = X_in.shape[1]

    fig = plt.figure()
    ax = fig.gca()
    cmap_in = plt.cm.gist_ncar
    cmap_out = plt.cm.jet
    title = "Inliers and the outliers in a " + str(D) + "-dimensional ambient space"
    sc = ax.scatter(X_in[:, 0], X_in[:, 1], cmap = cmap_in, label = "inliers")
    sc = ax.scatter(X_out[:, 0], X_out[:, 1], cmap = cmap_out, label = "outliers")
    plt.legend(loc = 1)
    plt.title(title)
    plt.show()
