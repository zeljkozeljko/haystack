import os
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import time
from itertools import islice


def unit_vector(vector):
    """ Normalises a vector  """
    return vector / np.linalg.norm(vector)

def sampling_ratios(N_in, d, N_out, D):
    print "Inlier sampling ratio =", float(N_in) / d
    print "Outlier sampling ratio =", float(N_out) / D
    print "The left hand side of (3.7) is " + str(float(N_in) / d) + ", while the right hand side is " + \
                str(np.sqrt(np.pi) * (np.sqrt(float(N_out) / (D - d)) + 1) ** 2)

def weights_update(X, P, delta):
    """ Updates the weights according to the rule in Algo 4.2 """
    weights = np.ones(X.shape[0])

    for i in range(len(weights)):
        weights[i] = 1.0 / max(delta, np.linalg.norm(X[i, :] - np.dot(P, X[i, :])))

    return weights

# X.shape[0] is just under the assumption that the vectors will be in rows of the matrix
# might change that though. Then also X[i, :] would need to be updated
# So, X.shape[0] is the number of vectors - i.e. data points
# X.shape[1] is the dimensionality - i.e. D

def weighted_least_squares(X, weights, dim):
    # Create the weighted covariance matrix
    C = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[0]):
        C += weights[i] * np.outer(X[i, :], X[i, :])

    U, S, V = np.linalg.svd(C)
    eta = np.zeros(X.shape[1])
    Srank = len(np.nonzero(S)[0])

    if np.allclose(S[int(np.floor(dim))], 0.0):
        for i in range(np.floor(dim)):
            eta[i] = 1.0
        eta[i + 1] = dim - np.floor(dim)
    else:
        for i in range(int(np.floor(dim)), len(eta)):
            # what if one of S[i]s is = 0?
            theta = (i - np.floor(dim)) / sum(S[:i] ** (-1) )
            if S[i] > theta and theta >= S[i + 1] or (i + 1) > Srank:
                break
        for i in range(len(eta)):
            if S[i] > theta:
                eta[i] = 1 - theta / S[i]
    # import pdb; pdb.set_trace()
    P = np.dot(U, np.dot(np.diag(eta), V))
    return P

def IRLS(X, d, delta, epsilon, spherical = "False"):

    if spherical == "True":
        for i in range(X.shape[0]):
            X[i, :] /= np.linalg.norm(X[i, :])
        print "hm?"
    error = {}
    weights = np.ones(X.shape[0])
    flag = "Failure"
    cnt = 0
    max_iterations = 1000
    error[cnt] = np.inf
    print "flag"

    while cnt < max_iterations:
        cnt += 1
        P = weighted_least_squares(X, weights, d)
        error[cnt] = 0

        for i in range(X.shape[0]):
            error[cnt] += weights[i] * np.linalg.norm(X[i, :] - np.dot(P, X[i, :])) ** 2

        if error[cnt] > error[cnt - 1] - epsilon:
            print error[cnt]
            print error[cnt - 1]
            flag = "Success"
            break

        weights = weights_update(X, P, delta)
    if flag == "Failure":
        print flag
    print "cnt is ", cnt
    return P
