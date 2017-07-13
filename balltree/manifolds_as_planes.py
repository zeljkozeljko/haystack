# coding: utf8
import numpy as np

""" General utilities """
def normal_nd(vectors):
    """
    Parameters
    -------------
    vectors : np.array, shape (space_dim, n_vectors)
        Matrix that contains columnwise vectors to which we want to search the
        orthogonal basis.

    Returns
    ------------
    Returns a matrix with space_dim - n_vectors columns, where each column is
    orthonogal to the given vectors and the matrix is orthonormal itself.
    """
    # import pdb; pdb.set_trace()
    U, S, V = np.linalg.svd(vectors.T, full_matrices = 1, compute_uv = 1)
    compl_basis_idx = np.where(S > 1e-14)[0]
    basis_idx = np.setdiff1d(np.arange(vectors.shape[0]), compl_basis_idx)
    return V.T[:, basis_idx] # columns of V to small S are orthogonal basis

""" Examples """
def swiss_roll(t1, t2, derivative = 0, repr_freq = 4.0, height = 6.0):
    fac = repr_freq * np.pi
    t1_n = fac * np.sqrt(t1)
    t2_n = height * t2
    # fac = 1
    # t1_n = fac * t1
    # t2_n = height * t2
    if derivative == 1:
        # Tangent vectors
        tan1 = np.array([fac * (np.cos(t1_n)/(2.0 * np.sqrt(t1)) - 0.5 * fac * np.sin(t1_n)),
                       fac * (np.sin(t1_n)/(2.0 * np.sqrt(t1)) + 0.5 * fac * np.cos(t1_n)),
                       0.0])
        # tan1 = np.array([fac * (np.cos(t1_n)/(2.0 * np.sqrt(t1)) - 0.5 * fac * np.sin(t1_n)),
        #                fac * (np.sin(t1_n)/(2.0 * np.sqrt(t1)) + 0.5 * fac * np.cos(t1_n)),
        #                0.0])
        # tan2 = np.array([0.0, 0.0, 1])
        tan2 = np.array([0.0, 0.0, height])
        return np.column_stack((tan1, tan2))
    elif derivative == -1:
        # Get normalised normal vector
        normals = normal_nd(swiss_roll(t1, t2, derivative = 1,
                                  repr_freq = repr_freq, height = height))
        return normals
    else:
        return np.array([t1_n * np.cos(t1_n), t1_n * np.sin(t1_n), t2_n])

def twin_peaks(t1, t2, derivative = 0, height = 1.0):
    if derivative == 1:
        # Tangent vectors
        tan1 = np.array([1.0, 0.0, height * (np.pi * np.cos(np.pi * t1) *
                                                            np.tanh(3.0 * t2))])
        tan2 = np.array([0.0, 1.0, 3.0 * height * np.sin(np.pi * t1) *
                                              np.square(1.0/np.cosh(3.0 * t2))])
        return np.column_stack((tan1, tan2))
    elif derivative == -1:
        # Get normalised normal vector
        normals = normal_nd(twin_peaks(t1, t2, derivative = 1,
                                  height = height))
        return normals
    else:
        return np.array([t1, t2, height * np.sin(np.pi * t1) * np.tanh(3.0 * t2)])


def inflate_the_roll(manifold_function, dimensions_to_add):
    def f_manifold(t, derivative = 0):
        if derivative == 0:
            ori_output = manifold_function(t, derivative)
            padded_zeros = np.zeros(dimensions_to_add)
            return np.append(ori_output, padded_zeros)
        elif derivative == 1:
            ori_output = manifold_function(t, derivative, dims_to_add = dimensions_to_add)
            #padded_zeros = np.zeros(dimensions_to_add)
            return ori_output
            # return np.append(ori_output, padded_zeros)
        else:
            ori_output = manifold_function(t, derivative = 1, dims_to_add = dimensions_to_add)
            # padded_zeros = np.zeros(dimensions_to_add)
            # output_padded = np.append(ori_output, padded_zeros)
            # import pdb; pdb.set_trace()
            return ori_output
            # return normal_nd(output_padded)
    return f_manifold

def swiss_rollv2(t, derivative = 0, dims_to_add = 0, repr_freq = 4.0, height = 6.0):
    fac = repr_freq * np.pi
    t1 = t[0]
    t2 = t[1]
    t1_n = fac * np.sqrt(t1)
    t2_n = height * t2
    if derivative == 1:
        # Tangent vectors
        # if dims_to_add == 0:
        #     tan1 = np.array([fac * (np.cos(t1_n)/(2.0 * np.sqrt(t1)) - 0.5 * fac * np.sin(t1_n)),
        #                    fac * (np.sin(t1_n)/(2.0 * np.sqrt(t1)) + 0.5 * fac * np.cos(t1_n)),
        #                    0.0])
        #     tan2 = np.array([0.0, 0.0, height])
        # else:
        tan1 = np.append(np.array([fac * (np.cos(t1_n)/(2.0 * np.sqrt(t1)) - 0.5 * fac * np.sin(t1_n)),
                       fac * (np.sin(t1_n)/(2.0 * np.sqrt(t1)) + 0.5 * fac * np.cos(t1_n)),
                       0.0]), np.zeros(dims_to_add) )
        tan2 = np.append(np.array([0.0, 0.0, height]), np.zeros(dims_to_add) )
            # import pdb; pdb.set_trace()
        return np.column_stack((tan1, tan2))
    elif derivative == -1:
        # Get normalised normal vector
        normals = normal_nd(swiss_rollv2(t, derivative = 1, dims_to_add = dims_to_add,
                                  repr_freq = repr_freq, height = height))
        return normals
    else:
        return np.append(np.array([t1_n * np.cos(t1_n), t1_n * np.sin(t1_n), t2_n]), np.zeros(dims_to_add) )

def sphere(n_points, dim = 3, R = 1, derivative = 0):

    points_on_sphere = np.random.randn(dim, n_points)
    points_on_sphere /= np.linalg.norm(points_on_sphere, axis = 0)

    if derivative == 1:
        tangents = {}
        for i in range(n_points):
            # import pdb; pdb.set_trace()
            tangents[i] = np.linalg.qr(np.eye(dim) - np.outer(points_on_sphere[:, i], points_on_sphere[:, i]))[0][:, :-1]
        return R * points_on_sphere, tangents
    elif derivative == -1:
        normals = {}
        tangents = {}
        for i in range(n_points):
            tangents[i] = np.linalg.qr(np.eye(dim) - np.outer(points_on_sphere[:, i], points_on_sphere[:, i]))[0][:, :-1]
            normals[i] = points_on_sphere[:, i]
        return R * points_on_sphere, tangents, normals
    else:
        return R * points_on_sphere
