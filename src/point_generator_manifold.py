# coding: utf8
import numpy as np
from scipy.integrate import quad
from scipy.optimize import newton


""" General utilities """
def curve_length(f_manifold, t0, tend):
    def integrand(t_running):
        derivatives = f_manifold(t_running, derivative = 1)
        return np.linalg.norm(derivatives, ord = 2)
    return quad(integrand, t0, tend)[0]

def curve_as_arclength(s, f_manifold, t0, derivative = 0):
    def root_problem(t_running):
        return curve_length(f_manifold, t0, t_running) - s
    def root_problem_prime(t_running):
        derivatives = f_manifold(t_running, derivative = 1)
        return np.linalg.norm(derivatives)
    root = newton(root_problem, 0, root_problem_prime)
    return f_manifold(root, derivative = derivative)

def normal_nd(vector):
    A = np.reshape(vector, (1, vector.shape[0]))
    U, S, V = np.linalg.svd(A, full_matrices = 1, compute_uv = 1)
    compl_basis_idx = np.where(S > 1e-14)[0]
    basis_idx = np.setdiff1d(np.arange(vector.shape[0]), compl_basis_idx)
    return V[:, basis_idx] # columns of V to small S are orthogonal basis

""" Curves wrap in high dim [a,b]->R^n where most of the dimensions are
    only noise."""
def inflate_ambient_space(manifold_function, dimensions_to_add):
    def f_manifold(t, derivative = 0):
        if derivative == 1 or derivative == 0:
            ori_output = manifold_function(t, derivative)
            padded_zeros = np.zeros(dimensions_to_add)
            return np.append(ori_output, padded_zeros)
        else:
            ori_output = manifold_function(t, derivative = 1)
            padded_zeros = np.zeros(dimensions_to_add)
            output_padded = np.append(ori_output, padded_zeros)
            return normal_nd(output_padded)
    return f_manifold

""" For 1D Curve """
def from_1d_parametrisation_with_basepoints(a, b, f_manifold, n_samples,
                                            noise_level):
    # Find length corresponding to end parameter b
    length = curve_length(f_manifold, a, b)
    s_disc = np.linspace(0.0, length, n_samples)
    # Point cloud directly ON the manifold
    point_for_shape = curve_as_arclength(s_disc[0], f_manifold, a)
    points_on_manifold = np.zeros((point_for_shape.shape[0], n_samples))
    for i in range(n_samples):
            points_on_manifold[:,i] = curve_as_arclength(s_disc[i], f_manifold, a)
    # Add some noise in normal direction
    points_with_noise = np.zeros(points_on_manifold.shape)
    for i in range(n_samples):
        normal_space = curve_as_arclength(s_disc[i], f_manifold, a,
                                          derivative = -1)
        # Draw n_normal_vectors random numbers to get a random vector in this
        # space and renormalise the result
        random_coefficients = np.random.uniform(-noise_level, noise_level,
                                                size = normal_space.shape[1])
        normal_vector = np.sum(normal_space * random_coefficients, axis=1)
        points_with_noise[:,i] = points_on_manifold[:,i] + normal_vector
    return points_on_manifold, points_with_noise
