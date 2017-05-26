# coding: utf8
import numpy as np

def sampling_ratios(N_in, d, N_out, D):
    print "Inlier sampling ratio =", float(N_in) / d
    print "Outlier sampling ratio =", float(N_out) / D
    print "The left hand side of (3.7) is " + str(float(N_in) / d) + ", while the right hand side is " + \
                str(np.sqrt(np.pi) * (np.sqrt(float(N_out) / (D - d)) + 1) ** 2)

def weights_update(X, P, delta):
    """ Updates the weights according to the rule in Algo 4.2 """
    n, D = X.shape # Number of points and number of features
    weights = np.ones(n)
    for i in range(n):
        #print delta, np.linalg.norm(X[i, :] - np.dot(P, X[i, :]))
        weights[i] = 1.0 / max(delta, np.linalg.norm(X[i, :] - np.dot(P, X[i, :])))
    return weights

# X.shape[0] is just under the assumption that the vectors will be in rows of the matrix
# might change that though. Then also X[i, :] would need to be updated
# So, X.shape[0] is the number of vectors - i.e. data points
# X.shape[1] is the dimensionality - i.e. D

def weighted_least_squares(X, weights, dim):
    # Create the weighted covariance matrix
    n, D = X.shape # Number of points and number of features
    d = np.floor(dim).astype('int')
    C = np.zeros((D, D))
    eta = np.zeros(D)
    # Compute Covariance matrix
    for i in range(n):
        C += weights[i] * np.outer(X[i, :], X[i, :])
    # Compute Eigenvalues
    S, U = np.linalg.eigh(C)
    # Resort eigenvalues to be non-increasing
    idx = S.argsort()[::-1]
    S = S[idx]
    U = U[:,idx]
    #Srank = len(np.nonzero(S)[0])
    # print "The last singular value S[d] = ", S[d]
    if S[d] == 0:
        # Setting eta for i < np.fooor(dim)
        for i in range(d):
            eta[i] = 1.0
        # Setting eta for np.fooor(dim) + 1
        eta[d + 1] = dim - d
    else:
        for i in range(d, D):
            theta = ((i + 1) - dim) / np.sum(S[:i] ** (-1) )
            print theta
            if i+1 >= D or (S[i] > theta and theta >= S[i + 1]):
                break
        for i in range(D):
            if S[i] > theta:
                eta[i] = 1 - theta / S[i]
    # print "Eigenvalues: ", S
    # print "Eta: ", eta
    # import pdb
    # pdb.set_trace()
    P = np.dot(U, np.dot(np.diag(eta), U.T))
    return P

def IRLS(X, d, delta, epsilon, spherical = "False"):
    n, D = X.shape # Number of points and number of features
    if spherical == "True":
        for i in range(n):
            X[i, :] /= np.linalg.norm(X[i, :])
        print "Made data spherical..."
    error = {}
    weights = np.ones(n)
    flag = "Failure"
    cnt = 0
    max_iterations = 1000
    error[cnt] = np.inf

    while cnt < max_iterations:
        cnt += 1
        P = weighted_least_squares(X, weights, d)
        error[cnt] = 0

        for i in range(n):
            error[cnt] += weights[i] * np.linalg.norm(X[i, :] - np.dot(P, X[i, :])) ** 2
        print np.abs(error[cnt] - error[cnt-1])
        if np.abs(error[cnt]-error[cnt - 1]) < 10e-12:
            print error
            flag = "Success"
            break
        weights = weights_update(X, P, delta)
    if flag == "Failure":
        print flag
    print "cnt is ", cnt
    return P
