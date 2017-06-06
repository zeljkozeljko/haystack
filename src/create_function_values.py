# coding: utf8
import numpy as np

def function_values_for_linear_space(points, space, f_manifold):
    """ Returns function values for points that belong to the given linear space.
    Evaluation works as follows:

        fval[i] = f_manifold(space.T.dot(space).dot(points[:,i]))
    """
    amb_dim, n_points = points.shape
    fval = np.zeros(n_points)
    for i in range(n_points):
        fval[i] = f_manifold(space.T.dot(points[:,i]))
    return fval
