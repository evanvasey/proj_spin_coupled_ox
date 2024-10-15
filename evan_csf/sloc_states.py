import numpy as np 
from RDM import get_pf_spin_operator
from MO_tools import mo_basis_change
from CSF_tools import round_coeff_zero,cumul_coeff






# in case no side localised functions are needed
def get_sloc_empty(hf_coeffs,lowdin_coeffs,overlap,grouped=False):
    return [],[]



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
    
    


# create semi localized states, containing a MO part and a CSF part
def get_semi_loc_states(core_mo_coeffs,det_core,CSF_dets,CSF_coeffs,lowdin_coeffs,overlap):
    core_dets,core_coeffs = mo_basis_change(det_core,core_mo_coeffs,lowdin_coeffs,overlap)
    core_dets,core_coeffs = round_coeff_zero(core_dets,core_coeffs,threshold=1e-10)
    list_dets = []
    list_coeffs = []
    for det_core,coeff_core in zip(core_dets,core_coeffs):
        for det_csf,coeff_csf in zip(CSF_dets,CSF_coeffs):
            # get new det. ATTENTION pf and zero because of double operator
            det = get_prod_dets(det_core,det_csf)
            if det==0:
                continue
            coeff = coeff_core*coeff_csf
            list_dets.append(det)
            list_coeffs.append(coeff)
            
    
    return list_dets,list_coeffs




# get the side localised states of the cyclic H4 molecule
def get_sloc_H4(hf_coeffs,lowdin_coeffs,overlap,grouped=False):

    core_mo_coeffs = np.zeros(hf_coeffs.shape)

    core_mo_coeffs[:,0] =  (hf_coeffs[:,0] + hf_coeffs[:,1])/np.sqrt(2)
    core_mo_coeffs[:,1] =  (hf_coeffs[:,0] - hf_coeffs[:,1])/np.sqrt(2)
    core_mo_coeffs[:,2] =  (hf_coeffs[:,0] + hf_coeffs[:,2])/np.sqrt(2)
    core_mo_coeffs[:,3] =  (hf_coeffs[:,0] - hf_coeffs[:,2])/np.sqrt(2)

    sloc_dets1,sloc_coeffs1 = get_semi_loc_states(core_mo_coeffs,[1.0,1,0,0,0,1,0,0,0],[[1.0,0,0,1,0,0,0,0,1],[1.0,0,0,0,1,0,0,1,0]],[1/np.sqrt(2),1/np.sqrt(2)],lowdin_coeffs,overlap) 
    sloc_dets1,sloc_coeffs1 = cumul_coeff(sloc_dets1,sloc_coeffs1,threshold=1e-10)
    sloc_dets2,sloc_coeffs2 = get_semi_loc_states(core_mo_coeffs,[1.0,0,1,0,0,0,1,0,0],[[1.0,1,0,0,0,0,1,0,0],[1.0,0,1,0,0,1,0,0,0]],[1/np.sqrt(2),1/np.sqrt(2)],lowdin_coeffs,overlap) 
    sloc_dets2,sloc_coeffs2 = cumul_coeff(sloc_dets2,sloc_coeffs2,threshold=1e-10)
    sloc_dets3,sloc_coeffs3 = get_semi_loc_states(core_mo_coeffs,[1.0,0,0,1,0,0,0,1,0],[[1.0,0,1,0,0,0,0,1,0],[1.0,0,0,1,0,0,1,0,0]],[1/np.sqrt(2),1/np.sqrt(2)],lowdin_coeffs,overlap) 
    sloc_dets3,sloc_coeffs3 = cumul_coeff(sloc_dets3,sloc_coeffs3,threshold=1e-10)
    sloc_dets4,sloc_coeffs4 = get_semi_loc_states(core_mo_coeffs,[1.0,0,0,0,1,0,0,0,1],[[1.0,1,0,0,0,0,0,0,1],[1.0,0,0,0,1,1,0,0,0]],[1/np.sqrt(2),1/np.sqrt(2)],lowdin_coeffs,overlap) 
    sloc_dets4,sloc_coeffs4 = cumul_coeff(sloc_dets4,sloc_coeffs4,threshold=1e-10)

    if grouped==False:
        return [sloc_dets1,sloc_dets2,sloc_dets3,sloc_dets4],[sloc_coeffs1,sloc_coeffs2,sloc_coeffs3,sloc_coeffs4]
    else:
        sloc_dets,sloc_coeffs = normalise_coeffs_det(sloc_dets1+sloc_dets2+sloc_dets3+sloc_dets4,sloc_coeffs1+sloc_coeffs2+sloc_coeffs3+sloc_coeffs4)
        return [sloc_dets],[sloc_coeffs]
