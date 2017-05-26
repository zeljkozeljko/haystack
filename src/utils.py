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
