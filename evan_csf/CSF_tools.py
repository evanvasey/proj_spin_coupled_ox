import numpy as np




### CSF permutation ###


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

# permute the determinants of a CSF according to a permutation pattern
def permute_csf(list_dets,permutation):
    list_dets_local = list_dets.copy()
    for i,det in enumerate(list_dets_local):
        new_det = permute_det(det,permutation)
        list_dets_local[i] = new_det
    return list_dets_local


### END ###




# Calculates the overlap of CSFs (dets,coeffs) in an orthonormal base.
def overlap_CSF(list_bra_dets,list_bra_coeffs,list_ket_dets,list_ket_coeffs):
    overlap = 0
    for bra,coeff_bra in zip(list_bra_dets,list_bra_coeffs):
        for ket,coeff_ket in zip(list_ket_dets,list_ket_coeffs):
            if bra[1:] == ket[1:]:
                overlap += coeff_ket*coeff_bra*bra[0]*ket[0]
    return overlap

# calculate overlap matrix between CSFs or linear combination of determinants in general
def overlap_matrix_CSF(list_states_dets,list_states_coeffs):
    n_states = len(list_states_dets)
    S = np.zeros((n_states,n_states))
    for i,(bra_dets,bra_coeffs) in enumerate(zip(list_states_dets,list_states_coeffs)):
        for j,(ket_dets,ket_coeffs) in enumerate(zip(list_states_dets,list_states_coeffs)):
            S[i,j] = overlap_CSF(bra_dets,bra_coeffs,ket_dets,ket_coeffs)
    return S


# in a list of coefficients and determinants remove the coeffs below a threshold and their corresponding determinant.
def round_coeff_zero(dets,coeffs,threshold):
    new_dets = []
    new_coeffs = []
    for det,coeff in zip(dets,coeffs):
        if abs(coeff) > threshold:
            new_dets.append(det)
            new_coeffs.append(coeff)
    return new_dets,new_coeffs

# cumulate the coefficients of same determinants in a list of determinants and coefficients
def cumul_coeff(det_list,coeff_list,threshold=None):
    det_dict = dict()
    # dictionary of every different occupation vector, for same occ. vector the coefficient times the pf are added to one another.
    for det,coeff in zip(det_list,coeff_list):
        occupation = tuple(det[1:])
        if occupation not in det_dict:
            det_dict[occupation] = 0
        det_dict[occupation] += det[0]*coeff 

    new_det_list = []
    new_coeff_list = []
    # creating the output with the new coeffs and only different occ. vectors.
    for occupation,coeff in det_dict.items():
        # since the pf has been included in the coeff we are left with only positive pf.
        new_det_list.append([1.0]+list(occupation))
        new_coeff_list.append(coeff)
    if threshold != None:
        new_det_list,new_coeff_list = round_coeff_zero(new_det_list,new_coeff_list,threshold)
    return new_det_list,new_coeff_list


# normalise coefficients of a list of determinants so that <psi|psi> = 1 if |psi> is the ket containing the determinants and their coeffs. 
# we start by cumulating same occ. vectors. The function can also round to zero the super small coefficients.
def normalise_coeffs_det(det_list,coeff_list,threshold=1e-8):
    det_list,coeff_list = cumul_coeff(det_list,coeff_list,threshold)
    N2 = 0
    for coeff in coeff_list:
        N2 += coeff**2
    return det_list,list(np.array(coeff_list)/np.sqrt(N2))





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
