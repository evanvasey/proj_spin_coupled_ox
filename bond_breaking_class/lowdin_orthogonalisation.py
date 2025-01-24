from pyscf import gto
import numpy as np





# The goal of this procedure is to obtain the symmetric orthogonalisation of the atomic orbitals
# this yields the orthogonal orbitals the closest to the AOs. These orbitals are called Lowdin orbitals


# diagonalise a matrix with eigenvalues, eigenvectors
def diag_matrix(matrix):
    eigenval,u = np.linalg.eigh(matrix)
    diag_matrix = np.diag(eigenval)
    uinv = np.linalg.inv(u)
    return u,diag_matrix,uinv

# calculate the inverse square root of a matrix
def inv_square_root_matrix(S):
    U,Sdiag,Uinv = diag_matrix(S)
    U = np.real(U)
    Sdiag = np.real(Sdiag)
    Uinv = np.real(Uinv)
    Sdiag_invsq = np.diag(1/np.sqrt(np.diag(Sdiag)))
    S_invsq = np.matmul(U,np.matmul(Sdiag_invsq,Uinv))
    return S_invsq

# obtain the Lowdin orbitals from the overlap matrix of AOs 
def get_symmetric_mo_coeffs(overlap_ao,permut=None):
    n_dim = overlap_ao.shape[0]
    if permut == None:
        identity = np.eye(n_dim)
    else:
        identity = permut
    inv_sq_overlap = inv_square_root_matrix(overlap_ao)
    symmetric_mo_coeffs = np.matmul(inv_sq_overlap,identity)
    return symmetric_mo_coeffs








