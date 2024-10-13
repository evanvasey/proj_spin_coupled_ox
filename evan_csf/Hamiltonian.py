import numpy as np




# FULL MOLECULAR HAMILTONIAN
def full_H(kets,coeffs,h_matrix,g_matrix,bras=None,bras_coeffs=None):
    n_dim = len(kets[0]) -1
    n_half = n_dim//2

    # in these RDMs the integral over spin coordinate is already taken into account, the double creation/annihilation operators too
    # get one and two spatial RDMs
    one_rdm = get_one_rdm_mc_fast(kets,coeffs,bras,bras_coeffs)
    one_rdm = get_spatial_one_rdm_fast(one_rdm)
    
    two_rdm = get_two_rdm_mc_fast(kets,coeffs,bras,bras_coeffs)
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
def get_H_CSF_matrix(list_CSF_dets,list_CSF_coeffs,h_matrix,g_matrix,energy_nuc):
    matrix_dim = len(list_CSF_dets)
    H_matrix = np.zeros((matrix_dim,matrix_dim))
    for i,(list_dets,list_coeffs) in enumerate(zip(list_CSF_dets,list_CSF_coeffs)):
        list_dets,list_coeffs = normalise_coeffs_det(list_dets,list_coeffs) 
        H_matrix[i,i] = full_H(list_dets,list_coeffs,h_matrix,g_matrix) + energy_nuc 
    for i,(list_dets_i,list_coeffs_i) in enumerate(zip(list_CSF_dets,list_CSF_coeffs)):
        for j,(list_dets_j,list_coeffs_j) in enumerate(zip(list_CSF_dets,list_CSF_coeffs)):
            if i==j:
                continue
            elif i < j:
                
                list_dets_i,list_coeffs_i = normalise_coeffs_det(list_dets_i,list_coeffs_i)
                list_dets_j,list_coeffs_j = normalise_coeffs_det(list_dets_j,list_coeffs_j)

                H_matrix[i,j] = full_H(list_dets_i,list_coeffs_i,h_matrix,g_matrix,list_dets_j,list_coeffs_j) + overlap_CSF(list_dets_i,list_coeffs_i,list_dets_j,list_coeffs_j)*energy_nuc

    for i,(list_dets_i,list_coeffs_i) in enumerate(zip(list_CSF_dets,list_CSF_coeffs)):
        for j,(list_dets_j,list_coeffs_j) in enumerate(zip(list_CSF_dets,list_CSF_coeffs)):
            if i==j:
                continue
            elif i > j:
                # matrix is symmetric since H is hermitian and we use real wavefunctions
                H_matrix[i,j] = H_matrix[j,i] 
    return H_matrix
