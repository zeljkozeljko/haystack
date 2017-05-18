import os
import numpy as np
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
    sc = ax.scatter(X_in[:, 0], X_in[:, 1], c = 'r', label = "inliers")
    sc = ax.scatter(X_out[:, 0], X_out[:, 1], c = 'b', label = "outliers")
    plt.legend(loc = 1)
    plt.title(title)
    plt.show()

def plot_points_and_projected(orthoproj, spherical_orthoproj, real_orthoproj, X_in, X_out):
    N_in = X_in.shape[0]
    N_out = X_out.shape[0]
    D = X_in.shape[1]
    fig, ax = plt.subplots(3,2)
    title = "Inliers and the outliers in a " + str(D) + "-dimensional ambient space"
    ax[0][0].scatter(X_in[:, 0], X_in[:, 1], c = 'r', label = "inliers")
    ax[0][0].scatter(X_out[:, 0], X_out[:, 1], c = 'b', label = "outliers")
    ax[0][1].scatter(real_orthoproj.dot(X_in.T).T[:, 0], real_orthoproj.dot(X_in.T).T[:, 1], c = 'r', label = "inliers")
    ax[0][1].scatter(real_orthoproj.dot(X_out.T).T[:, 0], real_orthoproj.dot(X_out.T).T[:, 1], c = 'b', label = "outliers")
    ax[1][0].imshow(orthoproj)
    ax[1][1].imshow(spherical_orthoproj)
    ax[2][0].scatter(orthoproj.dot(X_in.T).T[:, 0], orthoproj.dot(X_in.T).T[:, 1], c = 'r', label = "projected inliers")
    ax[2][0].scatter(orthoproj.dot(X_out.T).T[:, 0], orthoproj.dot(X_out.T).T[:, 1], c = 'b', label = "projected outliers")
    ax[2][1].scatter(spherical_orthoproj.dot(X_in.T).T[:, 0], spherical_orthoproj.dot(X_in.T).T[:, 1], c = 'r', label = "projected inliers")
    ax[2][1].scatter(spherical_orthoproj.dot(X_out.T).T[:, 0], spherical_orthoproj.dot(X_out.T).T[:, 1], c = 'b', label = "projected outliers")
    plt.title(title)
    plt.show()
