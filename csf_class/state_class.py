from CSF_tools import permute_vector,permute_det





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
        self.cumul_coeff(threshold)
        N2 = 0
        for coeff in self.coeffs:
            N2 += coeff**2
        self.coeffs = list(np.array(coeff_list)/np.sqrt(N2))

    # return the norm of the state
    def norm(self):
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
