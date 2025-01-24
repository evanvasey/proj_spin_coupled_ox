from pyscf import gto
import numpy as np









def diag_matrix(matrix):
    eigenval,u = np.linalg.eig(matrix)
    diag_matrix = np.diag(eigenval)
    uinv = np.linalg.inv(u)
    return u,diag_matrix,uinv

def inv_square_root_matrix(S):
    U,Sdiag,Uinv = diag_matrix(S)
    U = np.real(U)
    Sdiag = np.real(Sdiag)
    Uinv = np.real(Uinv)
    Sdiag_invsq = np.diag(1/np.sqrt(np.diag(Sdiag)))
    S_invsq = np.matmul(U,np.matmul(Sdiag_invsq,Uinv))
    return S_invsq

def get_symmetric_mo_coeffs(ao_overlap_matrix,permut=None):
    n_dim = ao_overlap_matrix.shape[0]
    if permut == None:
        identity = np.eye(n_dim)
    else:
        identity = permut
    inv_sq_overlap = inv_square_root_matrix(ao_overlap_matrix)
    symmetric_mo_coeffs = np.matmul(inv_sq_overlap,identity)
    return symmetric_mo_coeffs


def lowdin_ortho(list_orbitals,overlap_matrix):
    list_ortho_orbitals =[]
    T = inv_square_root_matrix(overlap)
    for i in list_orbitals:
        list_ortho_orbitals.append(np.matmul(T,i))
    return list_ortho_orbitals
