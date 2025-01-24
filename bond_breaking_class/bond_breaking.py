import numpy as np
from pyscf import scf
from scipy import linalg
from state import mc_state,merge_states
from lowdin_orthogonalisation import get_symmetric_mo_coeffs
from mo_tools import change_h_basis,change_g_basis,mo_basis_change,get_hf_coeffs,overlap_mo,overlap_matrix_mo
from mc_state_tools import overlap_CSF,overlap_matrix_CSF
from sloc_states import get_sloc_empty
from hamiltonian import full_H,get_H_matrix




# get the determinants in lowdin basis from dets in RHF basis
def get_RHF_states(RHF_states,hf_coeffs,lowdin_coeffs,overlap):
    states_dim = len(RHF_states)
    RHF_states_lowdin = []
    for i,state in enumerate(RHF_states):
        RHF_states_lowdin.append(mc_state(state.dets,state.coeffs))
        RHF_states_lowdin[i].change_basis(hf_coeffs,lowdin_coeffs,overlap)
    return RHF_states_lowdin


# main function that takes as inputs CSF states and RHF states, then outputs individual energies, L.C. energies for a range of bond distance
def get_Ecurve_CSF_RHF(r_array,RHF_states,CSF_states,molecule_func,HF_coeffs=None,number_sloc_states=None,sloc_grouped=False,get_sloc_func=get_sloc_empty,bond_length=None,savetxt=None):

    first = 1
    
    n_CSF = len(CSF_states)
    n_RHF = len(RHF_states)
    n_sloc = 0
    if number_sloc_states != None:
        n_sloc = number_sloc_states

    # alows to run a first RHF for a specific bond length that will then be used as an initial guess for the first bond length
    if bond_length != None:
        r_array=np.insert(r_array,0,bond_length)
    n_r = r_array.shape[0]


    # preparing output matrix that will go into files
    return_values_matrix = np.zeros((n_r,n_CSF+2+n_RHF+n_sloc))
    return_coeffs_matrix = np.zeros((n_r,n_CSF + n_RHF + 1+n_sloc))

    RHF_energy_list = []

    for i,r in enumerate(r_array):
        print(f"r = {r}")
        
        mol = molecule_func(r)
        
        overlap = mol.intor('int1e_ovlp')
        eri = mol.intor('int2e')
        kin = mol.intor('int1e_kin')
        vnuc = mol.intor('int1e_nuc')
        
        h = vnuc + kin
        g = eri
        
        lowdin_coeffs = get_symmetric_mo_coeffs(overlap)
        
        # change basis of h and g matrix to compute energy using only lowdin orbital basis
        h_lowdin = change_h_basis(h,lowdin_coeffs)
        g_lowdin = change_g_basis(g,lowdin_coeffs)
        
        hf = scf.RHF(mol)
        if first ==1:
            first = 0
        else:
            hf.mo_coeff = initial_guess_hf
        RHF_energy = hf.kernel()
        RHF_energy_list.append(RHF_energy)
        initial_guess_hf = hf.mo_coeff
        hf_coeffs = initial_guess_hf
        
        # allows to use our own HF orbitals rather than the ones given by PySCF
        # SHOULD WE USE PROJECTOR METHOD ON AO OR ON LOWDIN ???? 
        if HF_coeffs is not None:
            hf_coeffs = get_hf_coeffs(HF_coeffs,lowdin_coeffs,overlap)
            
        # transform the basis of RHF states into lowdin basis
        RHF_states_lowdin = get_RHF_states(RHF_states,hf_coeffs,lowdin_coeffs,overlap)
        
        
        # get the sloc states in lowdin basis
        sloc_states = get_sloc_func(hf_coeffs,lowdin_coeffs,overlap,sloc_grouped)
                                                                                                            
        # get the list of states for the hamiltonian matrix
        list_states = RHF_states_lowdin + CSF_states + sloc_states 


        # create the hamiltonian matrix
        H_matrix = get_H_matrix(list_states,h_lowdin,g_lowdin,mol.energy_nuc())
        
        # solve the generalised eigenvalue problem
        S = overlap_matrix_CSF(list_states)
        S[np.abs(S) < 1e-10] = 0
        S_inv = np.linalg.inv(S)
        S_inv_H_matrix = np.matmul(S_inv,H_matrix)
        eigenvalues,eigenvectors = np.linalg.eig(S_inv_H_matrix)
        eigenvalue_min = eigenvalues[0]
                        
        # select the eigenvector corresponding to the lowest eigenvalue
        for eigenvalue,eigenvector in zip(eigenvalues,eigenvectors.T):
            if eigenvalue <= eigenvalue_min:
                eigenvalue_min = eigenvalue
                eigenvector_min = eigenvector
       
        list_states_eigenvector = []  
        for state,ci in zip(list_states,eigenvector_min):
            list_states_eigenvector.append(mc_state(state.dets,list(np.array(state.coeffs)*np.real(ci))))
        eigenvector_state = merge_states(list_states_eigenvector) 
        eigenvector_state.normalize()
        energy_LC = full_H(eigenvector_state,h_lowdin,g_lowdin) + mol.energy_nuc() 
        
        diag_H_matrix = np.diag(H_matrix) 
                                                                                        
        return_coeffs_line = eigenvector_min
        return_coeffs_line = np.insert(return_coeffs_line,0,r)
        return_coeffs_matrix[i] = return_coeffs_line
                                                                                        
        return_values_line = np.insert(diag_H_matrix,0,energy_LC)
        return_values_line = np.insert(return_values_line,0,r)
        return_values_matrix[i] = return_values_line
    if bond_length != None:
        if savetxt !=None:
            np.savetxt(f"{savetxt}.data",return_values_matrix[1:,:])
            np.savetxt(f"{savetxt}_coeffs.data",return_coeffs_matrix[1:,:])
        return RHF_energy_list[1:]
    else:
        if savetxt !=None:
            np.savetxt(f"{savetxt}.data",return_values_matrix)
            np.savetxt(f"{savetxt}_coeffs.data",return_coeffs_matrix)
        return RHF_energy_list
