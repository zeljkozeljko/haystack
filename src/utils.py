import numpy as np

def extract_d_dim_principal_space(orthoprojector, dim):
    PS, PU = np.linalg.eigh(orthoprojector)
    # Resort eigenvalues to be non-increasing
    idx = PS.argsort()[::-1]
    PS = PS[idx]
    PU = PU[:,idx]
    PS = PS[:dim]
    PU = PU[:,:dim]
    return PU.dot(np.diag(PS)).dot(PU.T)

def calculate_gradient_outer_matrix(gradients):
    """ Calculates all outer gradient products and sums them up. This is called
    the outer gradient product matrix (GOM). Gradients are assumed to be in
    columns. """
    n_features, n_samples = gradients.shape
    gradient_outer_matrix = np.zeros((n_features, n_features))
    for i in range(n_samples):
        gradient_outer_matrix += np.outer(gradients[:,i], gradients[:,i].T)
    return 1.0/n_samples * gradient_outer_matrix

def extract_single_linearspace_prescribed_dim(points, dimension):
    """ Extract a single linear space of the prescribed dimension. The space
    will be returned in form of an ONB in the matrix columns."""
    n_features = points.shape[0]
    U, S, V = np.linalg.svd(points)
    return U[:,0:dimension]
