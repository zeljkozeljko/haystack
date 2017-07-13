# coding: utf8
import numpy as np
from manifolds_as_planes import sphere

def from_2d_parametrisation(x1, x2, y1, y2, f_manifold, f_on_manifold, n_samples,
                            noise_level, extra_dims  = 0):
    # Draw uniform at random for intervals [x1, x2] and [y1, y2]
    t1 = np.random.uniform(low = x1, high = x2, size = n_samples)
    t2 = np.random.uniform(low = y1, high = y2, size = n_samples)
    # Point cloud directly ON the manifold
    point_for_shape = f_manifold([t1[0], t2[0]], dims_to_add = extra_dims)
    points_on_manifold = np.zeros((point_for_shape.shape[0], n_samples))
    tangents = np.zeros((n_samples, point_for_shape.shape[0]))
    for i in range(n_samples):
            points_on_manifold[:,i] = f_manifold([t1[i], t2[i]], dims_to_add = extra_dims)
    # Get the function values
    function_values = np.zeros(n_samples)
    for i in range(n_samples):
        function_values[i] = f_on_manifold([t1[i], t2[i]])
    # Add some noise in normal direction
    for i in range(n_samples):
        normal_space = f_manifold([t1[i], t2[i]], derivative = -1, dims_to_add = extra_dims)
        tangents[i, :] = f_manifold([t1[i], t2[i]], derivative = 1, dims_to_add = extra_dims)[:, 0]
        # Draw n_normal_vectors random numbers to get a random vector in this
        # space and renormalise the result
        random_coefficients = np.random.uniform(-noise_level, noise_level,
                                                size = normal_space.shape[1])
        normal_vector = np.sum(normal_space * random_coefficients, axis=1)
        # import pdb; pdb.set_trace()
        points_on_manifold[:,i] = points_on_manifold[:,i] + normal_vector
    # import pdb; pdb.set_trace()
    return points_on_manifold, function_values, tangents
