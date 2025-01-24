import numpy as np
from rdm import get_one_rdm_mc_fast,get_two_rdm_mc_fast,get_spatial_one_rdm_fast,get_spatial_two_rdm_fast 
from mc_state_tools import overlap_CSF









# FULL MOLECULAR HAMILTONIAN
def full_H(ket,h_matrix,g_matrix,bra=None):
    n_dim = len(ket.dets[0]) -1
    n_half = n_dim//2

    ket_dets = ket.dets
    ket_coeffs = ket.coeffs
    
    if bra==None:
        bra_dets = None
        bra_coeffs = None
    else:
        bra_dets = bra.dets
        bra_coeffs = bra.coeffs
    # in these RDMs the integral over spin coordinate is already taken into account, the double creation/annihilation operators too
    # get one and two spatial RDMs
    one_rdm = get_one_rdm_mc_fast(ket_dets,ket_coeffs,bra_dets,bra_coeffs)
    one_rdm = get_spatial_one_rdm_fast(one_rdm)
    
    two_rdm = get_two_rdm_mc_fast(ket_dets,ket_coeffs,bra_dets,bra_coeffs)
    two_rdm = get_spatial_two_rdm_fast(two_rdm)

    E = 0
    E1 = 0
    E2 = 0
    for i in range(n_half):
        for j in range(n_half):
            # contribution from one electron operator
            E1 += one_rdm[i,j]*h_matrix[i,j]
            for k in range(n_half):
                for l in range(n_half):
                    # contribution from two electron operator
                    E2 += 0.5*two_rdm[i,j,k,l]*g_matrix[i,j,k,l]
    E = E1 + E2
    return E


# function to get a Hamiltonian matrix in terms of a list of states when given these states in the form of a list of dets and a list of coeffs.
def get_H_matrix(list_states,h_matrix,g_matrix,energy_nuc):
    matrix_dim = len(list_states)
    H_matrix = np.zeros((matrix_dim,matrix_dim))
    for i,state in enumerate(list_states):
        H_matrix[i,i] = full_H(state,h_matrix,g_matrix) + energy_nuc 
    for i,state_i in enumerate(list_states):
        for j,state_j in enumerate(list_states):
            if i==j:
                continue
            elif i < j:
                H_matrix[i,j] = full_H(state_i,h_matrix,g_matrix,state_j) + overlap_CSF(state_i,state_j)*energy_nuc
    for i in range(matrix_dim):
        for j in range(matrix_dim):
            if i==j:
                continue
            elif i > j:
                # matrix is symmetric since H is hermitian and we use real wavefunctions
                H_matrix[i,j] = H_matrix[j,i] 
    return H_matrix

