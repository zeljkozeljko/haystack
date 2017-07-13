import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import time

from manifolds_as_planes import *
from parameterised_manifolds import from_2d_parametrisation
from visualisation import plot_separated
from sklearn import cluster
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import BallTree

def LaplaceMatrix(X, gamma = "False"):
    L = np.zeros((X.shape[0], X.shape[0]))
    D = np.zeros(X.shape[0])
    if gamma == "False":
        pwdm = pairwise_distances(X.T)
        gamma = 0.5 * np.median(pwdm)
        print gamma
        if gamma < 1e-10:
            gamma = 1.0
            print "Kernel width is too small"
        # gamma = 1.0 / ( 2 * gamma ** 2)
    # print gamma

    for i in range(0, X.shape[0]):
        L[i, :] = np.exp( - gamma * np.linalg.norm(X[i, :] - X, axis = 1) ** 2 )
        D[i] = np.sum(L[i, :])
    for j in range(0, X.shape[0]):
        L[:, j] *= 1/np.sqrt(D[j])
    return L

def ConnectivityMatrix(X, gamma = 1):
    L = np.zeros((X.shape[1], X.shape[1]))
    D = np.zeros(X.shape[1])

    for i in range(0, X.shape[1]):
        for j in range(0, X.shape[1]):
            if i == j:
                continue
            else:
                L[i, j] = 1.0 /(np.linalg.norm(X[:, i] - X[:, j]) ** 2 )
    #     D[i] = np.sum(L[i, :])
    # for j in range(0, len(X)):
    #     L[:, j] *= 1/np.sqrt(D[j])
    return L

def whatever(x):
    return x[0]

# Set parameters
x1, x2 = 0.1, 0.2
y1, y2 = 0.1, 0.2
n_outliers = 100
n_inliers = 100
noise = 0.5
manifold_dim = 2
ambient_dim = 99

f_manifold = swiss_rollv2
f_on_manifold = whatever
outliers, _, _ = from_2d_parametrisation(x1, x2, y1, y2, f_manifold, f_on_manifold, n_outliers, noise, ambient_dim - manifold_dim)
outliers = np.random.randn(ambient_dim + 1, n_outliers)
inliers, _, _ = from_2d_parametrisation(x1, x2, y1, y2, f_manifold, f_on_manifold, n_inliers, 0.0, ambient_dim - manifold_dim)
inliers = (inliers.T - np.mean(inliers, axis = 1)).T
outliers = outliers * np.mean(np.linalg.norm(inliers, axis = 1))/np.mean(np.linalg.norm(outliers, axis = 1))
# plot_separated(inliers, outliers)

# import pdb; pdb.set_trace()
# import pdb; pdb.set_trace()
all_pts = np.hstack((inliers, outliers))
C = ConnectivityMatrix(all_pts)
eigzC = np.sort(np.linalg.eigvals(C))
gaps_C = np.abs(eigzC[:len(eigzC) - 1] - eigzC[1:])
gammas = np.linspace(0.0000001, 200.0/(noise * ambient_dim), 200)
conn = np.zeros(len(gammas))
pwdm = pairwise_distances(all_pts.T)
gamma = 0.5 * np.median(pwdm)

tree = BallTree(all_pts.T)
radii = np.linspace(0.02, 5, 30)
inliers_tree = np.zeros((len(radii), n_inliers))
outliers_tree = np.zeros((len(radii), n_outliers))


for j in range(len(radii)):
    for i in range(n_inliers):
        # import pdb; pdb.set_trace()
        inliers_tree[j, i] += np.float(tree.query_radius(inliers[:, i], r = radii[j], count_only = True))# / np.float(n_inliers)
    for k in range(n_outliers):
        outliers_tree[j, k] += np.float(tree.query_radius(outliers[:, k], r = radii[j], count_only = True))# / np.float(n_outliers)
# import pdb; pdb.set_trace()
#Means and STD
inlier_mean = np.mean(inliers_tree, axis = 1)
inlier_std = np.std(inliers_tree, axis = 1)
outlier_mean = np.mean(outliers_tree, axis = 1)
outlier_std = np.std(outliers_tree, axis = 1)

fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(211)
ax1.errorbar(radii, inlier_mean, inlier_std)
ax1.set_title('Inliers')
# plt.xlabel('Radius')
plt.ylabel('Pts in the ball')
ax2 = fig.add_subplot(212)
ax2.errorbar(radii, outlier_mean, outlier_std)
ax2.set_title('Outliers')
plt.xlabel('Radius')
plt.ylabel('Pts in the ball')
plt.show()
# plt.savefig('EqualiseddD100.pdf', bbox_inches='tight')

# import pdb; pdb.set_trace()
#
# print "---------------------"
# print gamma
# print 1.0 / (2 * gamma ** 2)
# print "---------------------"
# for i in range(len(gammas)):
#     # L = LaplaceMatrix(all_pts.T, gamma = gammas[i])
# # eigzL = np.sort(np.linalg.eigvals(L))[::-1]
# # gaps_L = np.abs(eigzL[:len(eigzL) - 1] - eigzL[1:])
#     spectral = cluster.SpectralClustering(n_clusters = 2, eigen_solver = 'arpack', affinity = "rbf", gamma = gammas[i])
#     B = spectral.fit_predict(all_pts.T)
#     conn[i] = np.float(min(sum(B > 0), sum( B == 0) )) / np.float(n_inliers)
# fig = plt.figure()
# plt.plot(gammas, np.array(conn))
#
# plt.show()
# import pdb; pdb.set_trace()
# in_B = all_pts[:, B != 0]
# out_B = all_pts[:, B == 0]
# n_clusters = len(set(B)) - (1 if -1 in labels else 0)

# import pdb; pdb.set_trace()
# plot_separated(inliers, outliers)
# plot_separated(in_B, out_B, title = 'Approximated Inliers and outliers')
# import pdb; pdb.set_trace()
