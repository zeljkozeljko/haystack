import os
import time
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np

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

def plot_error(errors_mean_normal, errors_std_normal,
               errors_mean_spherical, errors_std_spherical,
               ambient_dim, intrinsic_dim, inlier_points, outlier_points,
               outlier_spreads, inlier_spreads, repititions, sampling, method,
               y_index, preprocessing = None, cutoff = 0.0):
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    # fig.suptitle(r'No. Samples: {0}, Noise Levels: {1}, Repititions: {2}, $d$: {3}, $\sigma_w$: {4}, $\sigma_k$: {5}, $c_\lambda$ : {6}'.format(
    #     points_per_space_list, normal_noise_bounds, repititions, ls_dim,
    #     sigma_w, sigma_k, lambda_factor), fontsize=16)
    legend_entries = []
    if y_index == 0:
        # Plot against n_inlier
        for k, n_outlier in enumerate(outlier_points):
            for l, outlier_spread in enumerate(outlier_spreads):
                for w, inlier_spread in enumerate(inlier_spreads):
                    ax[0].errorbar(inlier_points, errors_mean_normal[:,k,w,l], errors_std_normal[:,k,w,l])
                    ax[1].errorbar(inlier_points, errors_mean_spherical[:,k,w,l], errors_std_spherical[:,k,w,l])
                    legend_entries.append((n_outlier, outlier_spread, inlier_spread))
        ax[0].set_xlabel(r'Inlier points', fontsize=16)
        ax[1].set_xlabel(r'Inlier points', fontsize=16)
        ax[0].set_ylabel(r'$\lambda_{max}(P - \hat{P})$', fontsize=16)
        ax[1].set_ylabel(r'$\lambda_{max}(P - \hat{P})$', fontsize=16)
        plt.legend(legend_entries, ncol = 2)
    elif y_index == 1:
        # Plot against n_outlier
        for k, n_inlier in enumerate(inlier_points):
            for l, outlier_spread in enumerate(outlier_spreads):
                for w, inlier_spread in enumerate(inlier_spreads):
                    ax[0].errorbar(outlier_points, errors_mean_normal[k,:,w,l], errors_std_normal[k,:,w,l])
                    ax[1].errorbar(outlier_points, errors_mean_spherical[k,:,w,l], errors_std_spherical[k,:,w,l])
                    legend_entries.append((n_inlier, outlier_spread, inlier_spread))
        ax[0].set_xlabel(r'Outlier points', fontsize=16)
        ax[1].set_xlabel(r'Outlier points', fontsize=16)
        ax[0].set_ylabel(r'$\lambda_{max}(P - \hat{P})$', fontsize=16)
        ax[1].set_ylabel(r'$\lambda_{max}(P - \hat{P})$', fontsize=16)
        plt.legend(legend_entries, ncol = 2)
    elif y_index == 2:
        # Plot against inlier spread
        for k, n_outlier in enumerate(outlier_points):
            for l, outlier_spread in enumerate(outlier_spreads):
                for w, inlier_point in enumerate(inlier_point):
                    ax[0].errorbar(inlier_spreads, errors_mean_normal[w,k,:,l], errors_std_normal[w,k,:,l])
                    ax[1].errorbar(inlier_spreads, errors_mean_spherical[w,k,:,l], errors_std_spherical[w,k,:,l])
                    legend_entries.append((n_outlier, outlier_spread, inlier_point))
        ax[0].set_xlabel(r'Inlier spread', fontsize=16)
        ax[1].set_xlabel(r'Inlier spread', fontsize=16)
        ax[0].set_ylabel(r'$\lambda_{max}(P - \hat{P})$', fontsize=16)
        ax[1].set_ylabel(r'$\lambda_{max}(P - \hat{P})$', fontsize=16)
        plt.legend(legend_entries, ncol = 2)
    elif y_index == 3:
        # Plot against outlier_spread
        for k, n_outlier in enumerate(outlier_points):
            for l, n_inlier in enumerate(inlier_points):
                for w, inlier_spread in enumerate(inlier_spreads):
                    ax[0].errorbar(outlier_spreads, errors_mean_normal[l,k,w,:], errors_std_normal[l,k,w,:])
                    ax[1].errorbar(outlier_spreads, errors_mean_spherical[l,k,w,:], errors_std_spherical[l,k,w,:])
                    legend_entries.append((n_outlier, n_inlier, inlier_spread))
        ax[0].set_xlabel(r'Outlier spread', fontsize=16)
        ax[1].set_xlabel(r'Outlier spread', fontsize=16)
        ax[0].set_ylabel(r'$\lambda_{max}(P - \hat{P})$', fontsize=16)
        ax[1].set_ylabel(r'$\lambda_{max}(P - \hat{P})$', fontsize=16)
        plt.legend(legend_entries, ncol = 2)
    plt.title('D = {0}, d = {1}, sampling = {2}, method = {3}, repititions = {4}, preprocessing = {5}, cutoff = {6}'.format(
        ambient_dim, intrinsic_dim, sampling, method, repititions, preprocessing, cutoff))
    plt.show()
