import numpy as np
from mc_state_tools import permute_vector,permute_det
from mo_tools import overlap_mo,order_permutation,mo_basis_change
import itertools





### CLASS ###
# creation of a class for multiconfigurational states

class mc_state:
    def __init__(self,dets,coeffs=None):
        self.dets = dets
        self.coeffs = coeffs if coeffs is not None else [1.0]
   
    # round the coefficient close to zero
    def round_coeffs(self,threshold=1e-10):
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
    def normalize(self,threshold=1e-10):
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
    def change_basis(self,mo1_basis_coeff,mo2_basis_coeff,ao_overlap,threshold=1e-10):
        new_dets = []
        new_coeffs = []
        dets = self.dets
        coeffs = self.coeffs
        for det,coeff in zip(dets,coeffs):
            det_temp,coeff_temp = mo_basis_change(det,mo1_basis_coeff,mo2_basis_coeff,ao_overlap)
            new_dets += det_temp
            new_coeffs += list(np.array(coeff_temp)*coeff)
        self.dets = new_dets
        self.coeffs = new_coeffs
        self.cumul(threshold)
    

# merge a list of states together into one state
def merge_states(list_states,threshold=1e-10):
    list_dets = []
    list_coeffs = []
    for state in list_states:
        list_dets += state.dets
        list_coeffs += state.coeffs
    new_state = mc_state(list_dets,list_coeffs)
    new_state.cumul(threshold)
    return new_state
            
