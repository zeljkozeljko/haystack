import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


""" Methods to visualise 3D stuff, such as a scattered point cloud, a manifold
function or learned normal spaces and centers. """

def handle_3D_plot():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    return fig, ax

def plot_manifold_function(t, manifold_function, *args):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    add_manifold_function(t, manifold_function, ax, *args)
    plt.title(r'Manifold $\mathcal{M}')
    plt.show()

def add_manifold_function(t, manifold_function, ax, *args):
    manifold_points = np.asarray([manifold_function(ti, *args) for
                                                            ti in sorted(t)])
    ax.plot(manifold_points[:,0],manifold_points[:,1],manifold_points[:,2],
            color='k', linewidth=3.0)

def add_manifold_as_surface(t, manifold_function, ax, *args):
    """
    Here t is 2 x n_samples matrix and contains the base points whereas the
    manifold function yields a z value for each t[:,i]. Note that not every
    manifold can be represented as this, so this plotting function can not
    always be used.

    Parameters
    ----------------
    t : np.array, shape (2 x n_samples)
        Contains the base points such that
        (t[:,i],t[:,j],manifold_function(t[:,i], t[:,j])) represent the points
        on the surface.

    manifold_function : Function handle that can be called with two base points
                        and yields the third (z)-coordinate.
    """
    t1_sorted = sorted(t[0,:])
    t2_sorted = sorted(t[1,:])
    XX, YY = np.meshgrid(t1_sorted, t2_sorted)
    Z = np.zeros(XX.shape)
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            Z[i,j] = manifold_function(XX[i,j], YY[i,j], *args)[2]
    ax.plot_surface(XX, YY, Z, alpha=0.2)




def plot_scatter(point_cloud, labels, xlow, xhigh, ylow, yhigh, zlow, zhigh):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    add_scattered_pointcloud(point_cloud, labels, ax)
    ax.set_xlim3d(xlow, xhigh)
    ax.set_ylim3d(ylow, yhigh)
    ax.set_zlim3d(zlow, zhigh)
    plt.title(r'Scatter plot of $X$')
    plt.show()

def add_scattered_pointcloud(point_cloud, labels, ax, label = None):
    if len(set(labels)) == 1:
        sc = ax.scatter(point_cloud[0, :], point_cloud[1, :], point_cloud[2, :],
                   c=labels[:])
        plt.colorbar(sc)
    else:
        # define the colormap
        cmap = plt.cm.hsv
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        # define the bins and normalize
        bounds = np.linspace(0,len(np.unique(labels)),
            len(np.unique(labels)) + 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        if label is not None:
            idx = [i for i in range(len(labels)) if labels[i] in label]
        else:
            idx = np.arange(point_cloud.shape[1])
        sc = ax.scatter(point_cloud[0, idx], point_cloud[1, idx],
                        point_cloud[2, idx], c=labels[idx], cmap = cmap,
                        norm = norm)
        plt.colorbar(sc)


def plot_separated(inliers, outliers, title = ""):
    fig, ax = handle_3D_plot()
    ax.scatter(inliers[0, :], inliers[1, :], inliers[2, :], label = 'Inliers')
    ax.scatter(outliers[0, :], outliers[1, :], outliers[2, :], label = 'Outliers')
    plt.legend()
