import numpy as np




### CSF permutation ###
# functions required in the class definition to have a permuted CSF

# permute a vector with a permutation pattern
def permute_vector(vector,permutation):
    initial_vector = vector.copy()
    vector_local = vector.copy()
    n_dim = len(vector)
    pf = 1
    for i in range(n_dim):
        i_index = permutation.index(i)
        i_occupation = initial_vector[i_index]
        elem_left = np.sum(vector_local[i:i_index])
        pf *= (-1)**(elem_left*i_occupation)
        vector_local.insert(i,i_occupation)
        del vector_local[i_index+1]
    return vector_local,pf

# permute a single determinant according to a permutation pattern
def permute_det(det,permutation):
    n_dim = len(det)-1
    n_half = n_dim//2 
    pf = det[0]
    vector_alpha = det[1:n_half+1]
    vector_beta = det[n_half+1:]
    vector_alpha_permuted,pf_alpha = permute_vector(vector_alpha,permutation)
    vector_beta_permuted,pf_beta = permute_vector(vector_beta,permutation)
    new_det = [pf*pf_alpha*pf_beta] + vector_alpha_permuted + vector_beta_permuted
    return new_det


### END ###


# Calculates the overlap of CSFs (dets,coeffs) in an orthonormal base.
def overlap_CSF(bra,ket):
    list_bra_dets,list_bra_coeffs = bra.dets,bra.coeffs
    list_ket_dets,list_ket_coeffs = ket.dets,ket.coeffs
    overlap = 0
    for det_bra,coeff_bra in zip(list_bra_dets,list_bra_coeffs):
        for det_ket,coeff_ket in zip(list_ket_dets,list_ket_coeffs):
            if det_bra[1:] == det_ket[1:]:
                overlap += coeff_ket*coeff_bra*det_bra[0]*det_ket[0]
    return overlap

# calculate overlap matrix between CSFs or linear combination of determinants in general
def overlap_matrix_CSF(list_states):
    n_states = len(list_states)
    S = np.zeros((n_states,n_states))
    for i,bra in enumerate(list_states):
        for j,ket in enumerate(list_states):
            S[i,j] = overlap_CSF(bra,ket)
    return S


# get new det from direct product of two states
def get_prod_dets(det1,det2):
    pf = det1[0]*det2[0]
    occ1 = det1[1:]
    occ2 = det2[1:]
    #VERIFY INDEXING OF np.argwhere
    occ_index = np.argwhere(det1[1:])
    # VERIFY THEORY WHIHC OCC CHANGE WHICH DISAPPEAR
    for index in occ_index[::-1]:
        if occ2[index[0]] == 1:
            return 0
        else:
            pf_op,spin = get_pf_spin_operator(index[0],occ2)
            pf *= pf_op
            occ2[index[0]] = 1 
            
     
    return [pf] + occ2
