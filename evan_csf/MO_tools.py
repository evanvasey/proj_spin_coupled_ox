import numpy as np
import itertools





# normalise coefficient MO coefficients expressed in terms of AO when given a AO overlap matrix
def get_norm_MO_coeffs(unorm_mo_coeffs,overlap_AO):
    n_mo = unorm_mo_coeffs.shape[1]
    norm_mo_coeffs = np.zeros((n_mo,n_mo))
    norm_mo_coeffs = norm_mo_coeffs.astype(np.float64)
    for i in range(n_mo):
        N2 = overlap_mo(unorm_mo_coeffs[:,i],unorm_mo_coeffs[:,i],overlap_AO)
        norm_mo_coeffs[:,i] = unorm_mo_coeffs[:,i]/np.sqrt(N2)
    return norm_mo_coeffs
 

# get the hf coefficient from the coefficient obtained from projection operator (symmetry)       
 # the coefficients are in orthonormal basis so lowdin basis
def get_hf_coeffs(coeffs,lowdin_coeffs,overlap,tol=1e-10):
     coeffs = coeffs.astype(np.float64)
     n_mo = coeffs.shape[1]
     hf_coeffs = np.zeros((n_mo,n_mo))
     for i in range(n_mo):
         hf_coeffs[:,i] = np.matmul(lowdin_coeffs,coeffs[:,i]) 
     hf_coeffs = get_norm_MO_coeffs(hf_coeffs,overlap)
     hf_coeffs[np.abs(hf_coeffs) < tol] = 0.0
     return hf_coeffs


# change basis of the matrix containing the 1 electron operators, refered to as h. It should contain vnuc and kin.
# the change of basis is done using the MO coefficients of the new base in terms of AO. The h matrix obtain with vnuc and kin from PySCF is obviously in AO terms too.
def change_h_basis(h_matrix,mo_coeffs_matrix):
    transpose_mo_coeffs_matrix = np.transpose(mo_coeffs_matrix)
    h_new_basis = np.matmul(transpose_mo_coeffs_matrix,np.matmul(h_matrix,mo_coeffs_matrix))
    return h_new_basis


# change basis of the matrix containing 2 electrons operator, refered to as g. It should contain eri.
# same as for h matrix.
def change_g_basis(g_matrix,mo_coeffs_matrix):
    m = mo_coeffs_matrix
    g = g_matrix
    g_new_basis = np.einsum('ia,jb,kc,ld,ijkl->abcd',m,m,m,m,g)
    return g_new_basis


# Calculates the overlap <mo1|mo2> where mo1,mo2 are expressed in AO basis
def overlap_mo(mo1_coeff,mo2_coeff,ao_overlap):
    overlap = np.matmul(np.matmul(mo1_coeff,ao_overlap),mo2_coeff)
    return overlap


# calculate overlap matrix between MOs
def overlap_matrix_mo(mo_coeffs,overlap_AO,mo_coeffs2=None):
    n_mo = mo_coeffs.shape[1]
    n_mo2 = n_mo

    if mo_coeffs2 is not None:
        n_mo2 = mo_coeffs2.shape[1]
    else:
        mo_coeffs2 = mo_coeffs

    mo_coeffs2
    overlap_matrix = np.zeros((n_mo,n_mo))
    for i in range(n_mo):
        for j in range(n_mo2):
            moi = mo_coeffs[:,i]
            moj = mo_coeffs2[:,j]
            overlap_matrix[i,j] = overlap_mo(moi,moj,overlap_AO)
    return overlap_matrix

# Change the basis of a determinant from basis mo1 to basis mo2
def order_permutation(permutation):
    dim = len(permutation)
    pf = 1
    ordered_permutation = []
    for i in range(dim):
        index_min = np.argmin(permutation[i:])
        value =permutation.pop(index_min+i)

        permutation.insert(i, value)
        pf *= (-1)**(index_min)
    return pf,permutation

# change basis from a MO basis to another MO basis (or Lowdin). For example from HF basis to Lowdin.
def mo_basis_change(det,mo1_basis_coeff,mo2_basis_coeff,ao_overlap):
    # represent MOs from the initial basis in terms of the second basis
    mo1_in_mo2_basis = np.zeros((mo2_basis_coeff.shape[1],mo1_basis_coeff.shape[1]))
    for col_index_mo1 in range(mo1_basis_coeff.shape[1]):
        mo1 = mo1_basis_coeff[:,col_index_mo1]
        for col_index_mo2 in range(mo2_basis_coeff.shape[1]):
            mo2 = mo2_basis_coeff[:,col_index_mo2]
            mo1_in_mo2_basis[col_index_mo2,col_index_mo1] = overlap_mo(mo1,mo2,ao_overlap) 

    pf = det[0]

    occupation_vector = det[1:]

    n_dim = len(occupation_vector)
    n_half = n_dim//2

    occupation_vector_alpha = occupation_vector[:n_half]
    alpha_el = np.sum(np.array(occupation_vector_alpha))

    occupation_vector_beta = occupation_vector[n_half:]
    beta_el = np.sum(np.array(occupation_vector_beta))

    empty_occupation_half = []
    for i in range(n_half):
        empty_occupation_half.append(0)

    det_list_alpha = []
    coeff_list_alpha = []
    det_list_beta = []
    coeff_list_beta = []
    
    # keep the MO coefficients corresponding to occupied orbitals for alpha/beta electrons
    alpha_mo2_basis_coeff = mo1_in_mo2_basis[:,np.array(occupation_vector_alpha)==1]
    beta_mo2_basis_coeff = mo1_in_mo2_basis[:,np.array(occupation_vector_beta)==1]

    # make sure the shape of the vector is correct even if there is only one orbital kept in the matrix
    if alpha_mo2_basis_coeff.ndim == 1:
        alpha_mo2_basis_coeff =alpha_mo2_basis_coeff.reshape(-1, 1)
    if beta_mo2_basis_coeff.ndim == 1:
        beta_mo2_basis_coeff =beta_mo2_basis_coeff.reshape(-1, 1)

    values = list(range(n_half))
    # create iteration object to loop over the possible permuatation of creation operators that come from the basis change. Each creation operator in the initial MO basis is expressed as a L.C. of creation operators in the new MO basis
    permutations_alpha = list(itertools.permutations(values, int(alpha_el)))
    permutations_beta = list(itertools.permutations(values, int(beta_el)))
    # create occupation vector for alpha electrons for every permuatation
    for perm in permutations_alpha:
        new_occupation = empty_occupation_half.copy()
        pf,perm_ordered = order_permutation(list(perm)) 
        for index in perm_ordered:
            new_occupation[index] = 1

        det_list_alpha.append(new_occupation.copy())
        coeff = 1
        for mo1_index,mo2_index in enumerate(perm):
            coeff *= alpha_mo2_basis_coeff[mo2_index,mo1_index] 
        coeff_list_alpha.append(pf*coeff)
    # create occupation vector for beta electrons for every permuatation
    for perm in permutations_beta:
        new_occupation = empty_occupation_half.copy()
        pf,perm_ordered = order_permutation(list(perm))
        for index in perm_ordered:
            new_occupation[index] = 1

        det_list_beta.append(new_occupation.copy())
        coeff = 1
        for mo1_index,mo2_index in enumerate(perm):
            coeff *= beta_mo2_basis_coeff[mo2_index,mo1_index]
        coeff_list_beta.append(pf*coeff)
    # create combination of alpha and beta occupation vector
    det_list = []
    coeff_list = []
    for det_alpha,coeff_alpha in zip(det_list_alpha,coeff_list_alpha):
        for det_beta,coeff_beta in zip(det_list_beta,coeff_list_beta):
            det_list.append([1.0]+det_alpha + det_beta)
            coeff_list.append(coeff_alpha*coeff_beta)
    
    return det_list,coeff_list
