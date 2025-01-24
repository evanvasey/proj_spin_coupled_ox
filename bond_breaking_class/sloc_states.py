import numpy as np 
from state import mc_state,merge_states
from rdm import get_pf_spin_operator
from mo_tools import mo_basis_change






# in case no side localised functions are needed
def get_sloc_empty(hf_coeffs,lowdin_coeffs,overlap,grouped=False):
    return []



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
def get_semi_loc_states(core_mo_coeffs,core,CSF,lowdin_coeffs,overlap):
    core.change_basis(core_mo_coeffs,lowdin_coeffs,overlap)
    core.round_coeffs()
    list_dets = []
    list_coeffs = []
    core_dets,core_coeffs = core.dets,core.coeffs
    CSF_dets,CSF_coeffs = CSF.dets,CSF.coeffs
    for det_core,coeff_core in zip(core_dets,core_coeffs):
        for det_csf,coeff_csf in zip(CSF_dets,CSF_coeffs):
            # get new det. ATTENTION pf and zero because of double operator
            det = get_prod_dets(det_core,det_csf)
            if det==0:
                continue
            coeff = coeff_core*coeff_csf
            list_dets.append(det)
            list_coeffs.append(coeff)
            
    sloc_state = mc_state(list_dets,list_coeffs) 
    return sloc_state




# get the side localised states of the cyclic H4 molecule
def get_sloc_H4(hf_coeffs,lowdin_coeffs,overlap,grouped=False):

    core_mo_coeffs = np.zeros(hf_coeffs.shape)

    core_mo_coeffs[:,0] =  (hf_coeffs[:,0] + hf_coeffs[:,1])/np.sqrt(2)
    core_mo_coeffs[:,1] =  (hf_coeffs[:,0] - hf_coeffs[:,1])/np.sqrt(2)
    core_mo_coeffs[:,2] =  (hf_coeffs[:,0] + hf_coeffs[:,2])/np.sqrt(2)
    core_mo_coeffs[:,3] =  (hf_coeffs[:,0] - hf_coeffs[:,2])/np.sqrt(2)

    core1 = mc_state([[1.0,1,0,0,0,1,0,0,0]])
    CSF1 = mc_state([[1.0,0,0,1,0,0,0,0,1],[1.0,0,0,0,1,0,0,1,0]],[1/np.sqrt(2),1/np.sqrt(2)])
    sloc1 = get_semi_loc_states(core_mo_coeffs,core1,CSF1,lowdin_coeffs,overlap) 
    sloc1.cumul(threshold=1e-10)

    core2 = mc_state([[1.0,0,1,0,0,0,1,0,0]])
    CSF2 = mc_state([[1.0,1,0,0,0,0,1,0,0],[1.0,0,1,0,0,1,0,0,0]],[1/np.sqrt(2),1/np.sqrt(2)])
    sloc2 = get_semi_loc_states(core_mo_coeffs,core2,CSF2,lowdin_coeffs,overlap) 
    sloc2.cumul(threshold=1e-10)
    
    core3 = mc_state([[1.0,0,0,1,0,0,0,1,0]])
    CSF3 = mc_state([[1.0,0,1,0,0,0,0,1,0],[1.0,0,0,1,0,0,1,0,0]],[1/np.sqrt(2),1/np.sqrt(2)])
    sloc3 = get_semi_loc_states(core_mo_coeffs,core3,CSF3,lowdin_coeffs,overlap) 
    sloc3.cumul(threshold=1e-10)
    
    core4 = mc_state([[1.0,0,0,0,1,0,0,0,1]])
    CSF4 = mc_state([[1.0,1,0,0,0,0,0,0,1],[1.0,0,0,0,1,1,0,0,0]],[1/np.sqrt(2),1/np.sqrt(2)])
    sloc4 = get_semi_loc_states(core_mo_coeffs,core4,CSF4,lowdin_coeffs,overlap) 
    sloc4.cumul(threshold=1e-10)
    

    if grouped==False:
        return [sloc1,sloc2,sloc3,sloc4]
    else:
        sloc = merge_states([sloc1,sloc2,sloc3,sloc4])
        sloc.normalize(threshold=1e-10)
        return sloc
