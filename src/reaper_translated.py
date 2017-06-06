# coding: utf8
import numpy as np

__tolerance_eigv__ = 1e-10
__delta_reaper__ = 1e-10
__max_iter_reaper__ = 2000
__eps_reaper__ = 1e-15

def weights_update(X, Q, delta):
    n, D = X.shape # Number of points and number of features
    weights = np.ones(n)
    for i in range(n):
        #print delta, np.linalg.norm(X[i, :] - np.dot(P, X[i, :]))
        weights[i] = 1.0 / max(delta, np.linalg.norm(np.dot(Q, X[i, :])))
    return weights

def irls_procedure(X, weights, d):
    n, D = X.shape # Number of points and number of features
    # Compute Covariance matrix
    C = np.zeros((D, D))
    for i in range(n):
        C += weights[i] * np.outer(X[i, :], X[i, :])
    # Compute Eigenvalues
    S, U = np.linalg.eigh(C)
    # Resort eigenvalues to be non-increasing
    idx = S.argsort()[::-1]
    S = S[idx]
    U = U[:,idx]

    estimated_rank = np.sum(S > __tolerance_eigv__)
    if estimated_rank < d:
        return U[:,d:D].dot(U[:, d:D].T), 0
    else:
        S = np.maximum(S, __tolerance_eigv__ * np.ones(D))
        S_inv = np.reciprocal(S[::-1])
        for i in range(D - d + 1):
            theta = (D - d - i)/np.sum(S_inv[i:])
            if theta <= 1.0/S_inv[i]:
                break
        new_evs = np.minimum(np.ones(S.shape), theta*S_inv[::-1])
        Q = U.dot(np.diag(new_evs)).dot(U.T)
        alpha = np.sqrt(np.sum(np.square(Q.dot(X.T)), axis = 0))
        return Q, alpha

def reaper_matlab(X, d, P_init = None, spherical = "False", verbose = False):
    n, D = X.shape # Number of points and number of features
    dim = np.floor(d).astype('int')
    alpha_old = np.ones(n) * np.inf
    if spherical == "True":
        for i in range(n):
            X[i, :] /= np.linalg.norm(X[i, :])
        if verbose:
            print "Made data spherical..."
    if P_init is not None:
        Q = np.eye(P_init.shape) - P
    else:
        Q = (1.0 - np.float(d)/np.float(D)) * np.eye(D)
    ctr = 0
    while ctr < __max_iter_reaper__:
        weights = weights_update(X, Q, __delta_reaper__)
        Q, alpha = irls_procedure(X, weights, d)
        if all(alpha - alpha_old) < __eps_reaper__:
            if verbose:
                print "Converged to solution"
                return np.eye(D) - Q
        if verbose:
            print "Ctr: {0} : Max error : {1}".format(ctr, np.max(alpha-alpha_old))
        ctr += 1
        alpha_old = alpha
    else:
        print "Not reached convergence"
    return np.eye(D) - Q
