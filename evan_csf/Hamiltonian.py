import numpy as np
from RDM import get_one_rdm_mc_fast,get_two_rdm_mc_fast,get_spatial_one_rdm_fast,get_spatial_two_rdm_fast 
from Lowdin_Orthogonalisation import get_symmetric_mo_coeffs
from MO_tools import change_h_basis,change_g_basis,mo_basis_change,get_hf_coeffs
from CSF_tools import normalise_coeffs_det,overlap_CSF,overlap_matrix_CSF
from sloc_states import get_sloc_empty
from pyscf import scf




# energy of H atom in sto-3G with pyscf intor methods
H_energy = -0.46658185






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


# from a list of states composed of dets and coeffs we get their individual energies, the coefficients from the diagonalisation of the matrix and the energy of the L.C. of the states using the coefficients
def get_energy_from_matrix(list_states_dets,list_states_coeffs,h_lowdin,g_lowdin,energy_nuc):
    H_matrix = get_H_CSF_matrix(list_states_dets,list_states_coeffs,h_lowdin,g_lowdin,energy_nuc)
    eigenvalues,eigenvectors = np.linalg.eigh(H_matrix)
    dets_list = []
    coeffs_list = []
    for dets,coeffs,ci in zip(list_states_dets,list_states_coeffs,eigenvectors[:,0]):
        dets_list += dets
        coeffs_list += list(np.array(coeffs)*ci)
    dets_list,coeffs_list = normalise_coeffs_det(dets_list,coeffs_list)

    energy_LC = full_H(dets_list,coeffs_list,h_lowdin,g_lowdin) + energy_nuc

    diag_H_matrix = np.diag(H_matrix) + energy_nuc
    return energy_LC,eigenvectors[:,0],diag_H_matrix


# get the determinants in lowdin basis from dets in RHF basis, if there are multiple RHF determinants we cumulate them into one state
def get_RHF_state(RHF_states,hf_coeffs,lowdin_coeffs,overlap):
    RHF_dets_list = []
    RHF_coeffs_list = []
    for det in RHF_states:
        RHF_dets, RHF_coeffs = mo_basis_change(det,hf_coeffs,lowdin_coeffs,overlap)
                                                                                                                      
        RHF_dets, RHF_coeffs = normalise_coeffs_det(RHF_dets,RHF_coeffs,threshold=1e-10)
        RHF_dets_list += RHF_dets
        RHF_coeffs_list += RHF_coeffs
    RHF_dets_list,RHF_coeffs_list = normalise_coeffs_det(RHF_dets_list,RHF_coeffs_list,threshold=1e-10)
    return [RHF_dets_list],[RHF_coeffs_list]

# get the determinants in lowdin basis from dets in RHF basis
def get_RHF_states(RHF_states,hf_coeffs,lowdin_coeffs,overlap):
    RHF_dets_list = []
    RHF_coeffs_list = []
    for det in RHF_states:
        RHF_dets, RHF_coeffs = mo_basis_change(det,hf_coeffs,lowdin_coeffs,overlap)
                                                                                                                      
        RHF_dets, RHF_coeffs = normalise_coeffs_det(RHF_dets,RHF_coeffs,threshold=1e-10)
        RHF_dets_list.append(RHF_dets)
        RHF_coeffs_list.append(RHF_coeffs)
    return RHF_dets_list,RHF_coeffs_list


# MAYBE ADD OPTION TO PUT HF COEFF AS INPUT UNORM, ADD NUMBER OF SLOC AS INPUT VAR AND FUNCTION TO GET THE SLOC DETS/COEFFS
# main function that takes as inputs CSF states and RHF states, then outputs individual energies, L.C. energies for a range of bond distance
def get_Ecurve_CSF_RHF(r_array,RHF_states,list_CSF_dets,list_CSF_coeffs,molecule_func,HF_coeffs=None,number_sloc_states=None,sloc_grouped=False,get_sloc_func=get_sloc_empty,RHF_states_grouped=False,bond_length=None,savetxt=None):
    n_CSF = len(list_CSF_dets)
    n_RHF = len(RHF_states)
    n_sloc = 0
    RHF_change_basis_func = get_RHF_states
    if RHF_states_grouped:
        n_RHF = 1
        RHF_change_basis_func = get_RHF_state
    if number_sloc_states != None:
        n_sloc = number_sloc_states

    RHF_energy_list = []


    # alows to run a first RHF for a specific bond length that will then be used as an initial guess for the first bond length
    if bond_length != None:
        r_array=np.insert(r_array,0,bond_length)

    n_r = r_array.shape[0]

    first = 1

    return_values_matrix = np.zeros((n_r,n_CSF+2+n_RHF+n_sloc))

    return_coeffs_matrix = np.zeros((n_r,n_CSF + n_RHF + 1+n_sloc))

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
        if HF_coeffs is not None:
            hf_coeffs = get_hf_coeffs(HF_coeffs,lowdin_coeffs,overlap)
            
        print(lowdin_coeffs) 
        print(overlap)
        # to save cube files that allows to visualise the orbitals with VMD
        #tools.cubegen.orbital(mol, 'rhf_1.cube', hf.mo_coeff[:, 0]) 
        #tools.cubegen.orbital(mol, 'rhf_2.cube', hf.mo_coeff[:, 1]) 
        #tools.cubegen.orbital(mol, 'rhf_3.cube', hf.mo_coeff[:, 2]) 
        #tools.cubegen.orbital(mol, 'rhf_4.cube', hf.mo_coeff[:, 3]) 
                                                                                                            
                                                                                                            
        #tools.cubegen.orbital(mol, 'sym_1.cube', hf_coeff[:,0]) 
        #tools.cubegen.orbital(mol, 'sym_2.cube', hf_coeff[:,1]) 
        #tools.cubegen.orbital(mol, 'sym_3.cube', hf_coeff[:,2]) 
        #tools.cubegen.orbital(mol, 'sym_4.cube', hf_coeff[:,3]) 
        
        # transform the basis of RHF states into lowdin basis
        RHF_dets_list,RHF_coeffs_list = RHF_change_basis_func(RHF_states,hf_coeffs,lowdin_coeffs,overlap)
        print(RHF_dets_list,RHF_coeffs_list)
        
        # get the states semi localised (core and csf)
        sloc_states_dets,sloc_states_coeffs = get_sloc_func(hf_coeffs,lowdin_coeffs,overlap,sloc_grouped)
                                                                                                            
        # get the list of states for the hamiltonian matrix
        dets_list_matrix = RHF_dets_list + list_CSF_dets + sloc_states_dets 
        coeffs_list_matrix = RHF_coeffs_list + list_CSF_coeffs + sloc_states_coeffs 
                                                                                                            
        # create the hamiltonian matrix and solve the generalised eigenvalue problem
        H_matrix = get_H_CSF_matrix(dets_list_matrix,coeffs_list_matrix,h_lowdin,g_lowdin,mol.energy_nuc())
        S = overlap_matrix_CSF(dets_list_matrix,coeffs_list_matrix)
        S[np.abs(S) < 1e-10] = 0
        S_inv = np.linalg.inv(S)
        S_inv_H_matrix = np.matmul(S_inv,H_matrix)
        eigenvalues,eigenvectors = np.linalg.eig(S_inv_H_matrix)
        eigenvalue_min = eigenvalues[0]
                                                                                                            
        # select the eigenvector corresponding to the lowest eigenvalue
        print(eigenvalues)
        for eigenvalue,eigenvector in zip(eigenvalues,eigenvectors.T):
            if eigenvalue <= eigenvalue_min:
                eigenvalue_min = eigenvalue
                eigenvector_min = eigenvector
                                                                                                            
        dets_list = []
        coeffs_list = []
        for dets,coeffs,ci in zip(dets_list_matrix,coeffs_list_matrix,eigenvector_min):
            dets_list += dets 
            coeffs_list += list(np.array(coeffs)*np.real(ci))
        dets_list,coeffs_list = normalise_coeffs_det(dets_list,coeffs_list)
        energy_LC = full_H(dets_list,coeffs_list,h_lowdin,g_lowdin) + mol.energy_nuc() 
        
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
