import numpy as np
from CSF_tools import permute_vector,permute_det
import itertools



### FUNCTIONS FOR CSF PERMUTE ###

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

### END FUNCTIONS CSF PERMUTE ###



### FUNCTIONS FOR BASIS TRANSFORM ###

# Calculates the overlap <mo1|mo2> where mo1,mo2 are expressed in AO basis
def overlap_mo(mo1_coeff,mo2_coeff,ao_overlap):
    overlap = np.matmul(np.matmul(mo1_coeff,ao_overlap),mo2_coeff)
    return overlap



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
    print(det,mo1_basis_coeff,mo2_basis_coeff,ao_overlap)
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


### END FUNCTIONS BASIS TRANSFORM ###





class mc_state:
    def __init__(self,dets,coeffs):
        self.dets = dets
        self.coeffs = coeffs
   
    # round the coefficient close to zero
    def round_coeffs(self,threshold):
        new_dets = []
        new_coeffs = []
        for det,coeff in zip(self.dets,self.coeffs):
            if abs(coeff) > threshold:
                new_dets.append(det)
                new_coeffs.append(coeff)
        self.dets = new_dets
        self.coeffs = new_coeffs
    
    # cumulate the coefficients of same determinants
    def cumul(self,threshold=None):
        det_dict = dict()
        # dictionary of every different occupation vector, for same occ. vector the coefficient times the pf are added to one another.
        for det,coeff in zip(self.dets,self.coeffs):
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
        self.dets = new_det_list
        self.coeffs =new_coeff_list
        if threshold != None:
            self.round_coeffs(threshold)

    # normalize the coefficients of the determinants
    def normalize(self,threshold):
        self.cumul(threshold)
        N2 = 0
        for coeff in self.coeffs:
            N2 += coeff**2
        self.coeffs = list(np.array(self.coeffs)/np.sqrt(N2))

    # return the norm of the state
    def norm(self,threshold=1e-10):
        self.cumul(threshold)
        overlap = 0
        for bra,coeff_bra in zip(self.dets,self.coeffs):
            for ket,coeff_ket in zip(self.dets,self.coeffs):
                if bra[1:] == ket[1:]:
                    overlap += coeff_ket*coeff_bra*bra[0]*ket[0]
        return overlap
    # permute the determinants of a CSF according to a permutation pattern
    def permute(self,permutation):
        list_dets_local = self.dets.copy()
        for i,det in enumerate(list_dets_local):
            new_det = permute_det(det,permutation)
            list_dets_local[i] = new_det
        self.dets = list_dets_local

    # transform the basis of a state
    def change_basis(self,mo1_basis_coeff,mo2_basis_coeff,ao_overlap,threshold):
        new_dets = []
        new_coeffs = []
        for det,coeff in zip(self.dets,self.coeffs):

            det_temp,coeff_temp = mo_basis_change(det,mo1_basis_coeff,mo2_basis_coeff,ao_overlap)
            print(det_temp,coeff_temp,"step1")
            new_dets += det_temp
            new_coeffs += list(np.array(coeff_temp)*coeff)
        self.dets = new_dets
        self.coeffs = new_coeffs
        self.cumul(threshold)
    
            


            
