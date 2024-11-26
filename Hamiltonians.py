import numpy as np
import itertools
import time
import matplotlib.pyplot as plt
from pyscf import gto,scf,symm,tools
from ReducedDensityMatrices.ReducedDensityMatrices import get_one_rdm, get_two_rdm,get_spatial_one_rdm,get_spatial_two_rdm
from ReducedDensityMatrices.ReducedDensityMatrices import get_mc_one_rdm,get_mc_two_rdm,get_spatial_one_rdm,get_spatial_two_rdm
from CSF import CSF
from Lowdin_Orthogonalisation import get_symmetric_mo_coeffs 
from CSF_permutation import permute_vector,permute_csf  
from RDM import get_pf_spin_operator,get_two_rdm_fast,get_two_rdm_mc_fast,get_spatial_two_rdm_fast,get_one_rdm_mc_fast,get_spatial_one_rdm_fast,check_rdm



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




# normalise coefficient MO coefficients expressed in terms of AO when given a AO overlap matrix
def get_norm_MO_coeffs(unorm_mo_coeffs,overlap_AO):
    print(unorm_mo_coeffs.dtype)
    n_mo = unorm_mo_coeffs.shape[1]
    norm_mo_coeffs = np.empty((n_mo,n_mo))
    norm_mo_coeffs = norm_mo_coeffs.astype(np.float64)
    print(norm_mo_coeffs.dtype)
    for i in range(n_mo):
        N2 = overlap_mo(unorm_mo_coeffs[:,i],unorm_mo_coeffs[:,i],overlap_AO)
        print(N2)
        print(unorm_mo_coeffs[:,i])
        norm_mo_coeffs[:,i] = unorm_mo_coeffs[:,i].astype(np.float64)*(-1)/np.sqrt(N2).astype(np.float64)
    print(norm_mo_coeffs)
    return norm_mo_coeffs
        
    #transpose_unorm_mo_coeffs = np.transpose(unorm_mo_coeffs)

    #overlap_mo = np.matmul(transpose_unorm_mo_coeffs,np.matmul(overlap_matrix_AO,unorm_mo_coeffs))
    #norm_vector = 1/np.sqrt(np.diagonal(overlap_mo))
    #print(norm_vector,"norm vector")
    #print(unorm_mo_coeffs,"unorm mo coeffs")
    #norm_mo_coeffs = unorm_mo_coeffs*norm_vector
    #print(norm_mo_coeffs,"norm mo coeffs")
    #return norm_mo_coeffs

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


# get the list of lists of determinants and coefficients corresponding to different CSFs.
# UNDER CONSTRUCTION, SHOULD HAD FLEXIBILITY IN THE ACTIVE SPACE ETC.
def get_list_csf(list_S, list_coupling,mol,mo_coeffs):
    csf_list_dets = []
    csf_list_coeffs = []
    for S,coupling in zip(list_S,list_coupling):
        csf = CSF(mol, mo_coeffs, S, [], [0, 1, 2, 3], active_space=(4, 4), csf_build="genealogical", g_coupling=coupling)
        csf_list_dets = csf_list_dets + csf.dets
        csf_list_coeffs = csf_list_coeffs + csf.coeffs
    return csf_list_dets,csf_list_coeffs

# Calculates the overlap <mo1|mo2> where mo1,mo2 are expressed in AO basis
def overlap_mo(mo1_coeff,mo2_coeff,ao_overlap):
    overlap = np.matmul(np.matmul(mo1_coeff,ao_overlap),mo2_coeff)
    return overlap

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

# calculate overlap matrix between MOs
def overlap_matrix_mo(mo_coeffs,overlap_AO,mo_coeffs2=None):
    n_mo = mo_coeffs.shape[1]
    n_mo2 = n_mo

    if mo_coeffs2 != None:
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
   


# FUNCTIONS TO GET THE MOL OBJECT FROM PySCF FOR DIFFERENT MOLECULAR SYSTEMS

def H2_mol(r):
    mol = gto.Mole()
    mol.build(
            atom = f'''H 0 0 0; H 0 0 {r}''',
            basis = 'sto-3g',
            symmetry= True,
            spin = 0,
            charge = 0)

    return mol

def H4_mol(r):
    mol = gto.Mole()
    mol.build(
            atom = f'''H 0 0 0; H 0 {r} 0; H 0 {r} {r}; H 0 0 {r}''',
            basis = 'sto-3g',
            symmetry= True,
            spin=0,
            charge = 0)

    return mol 

def H6_mol(r):
    mol = gto.Mole()
    mol.build(
            atom = f'''H 0 {r} 0; H 0 {-r} 0; H 0 {r/2} {r*np.sqrt(3)/2}; H 0 {-r/2} {r*np.sqrt(3)/2};H 0 {-r/2} {-r*np.sqrt(3)/2};H 0 {r/2} {-r*np.sqrt(3)/2}''',
            basis = 'sto-3g',
            symmetry= True,
            charge = 0)
    return mol

def H2_H_mol(r):
    h2_bond_length = 0.74 
    if r > 0:
        r1 = -h2_bond_length
        r2 =  h2_bond_length + r
    elif r==0:
        r1 =  h2_bond_length 
        r2 =  h2_bond_length
    else:
        r1 =  h2_bond_length + r
        r2 = -h2_bond_length
    mol = gto.Mole()
    mol.build(
            atom = f'''H 0 0 0; H 0 0 {r1}; H 0 0 {r2}''',
            basis = 'sto-3g',
            symmetry= False,
            spin = 0,
            charge = 0)
                                              
    return mol




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
# UNDER CONSTRUCTION
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

def get_sloc_empty(hf_coeffs,lowdin_coeffs,overlap,grouped=False):
    return [],[]


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

# get the hf coefficient from the coefficient obtained from projection operator (symmetry)
# the coefficients are in orthonormal basis so lowdin basis
def get_hf_coeffs(coeffs,lowdin_coeffs,overlap,tol=1e-10):
    coeffs = coeffs.astype(np.float64)
    n_mo = coeffs.shape[1]
    hf_coeffs = np.zeros((n_mo,n_mo))
    for i in range(n_mo):
        print(np.matmul(lowdin_coeffs,coeffs[:,i]))
        hf_coeffs[:,i] = np.matmul(lowdin_coeffs,coeffs[:,i]) 
    print(hf_coeffs,"before normalisation")
    hf_coeffs = get_norm_MO_coeffs(hf_coeffs,overlap)
    hf_coeffs[hf_coeffs < tol] = 0.0
    return hf_coeffs



# MAYBE ADD OPTION TO PUT HF COEFF AS INPUT UNORM, ADD NUMBER OF SLOC AS INPUT VAR AND FUNCTION TO GET THE SLOC DETS/COEFFS
# main function that takes as inputs CSF states and RHF states, then outputs individual energies, L.C. energies for a range of bond distance
def get_Ecurve_CSF_RHF(r_array,RHF_states,list_CSF_dets,list_CSF_coeffs,molecule_func,HF_coeffs=None,number_sloc_states=None,sloc_grouped=False,get_sloc_func=None,RHF_states_grouped=False,bond_length=None,savetxt=None):
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
            #hf_coeffs = get_norm_MO_coeffs(HF_coeffs,overlap)
            #print(hf_coeffs,"hf coeffs")
            hf_coeffs = get_hf_coeffs(HF_coeffs,lowdin_coeffs,overlap)
            print(hf_coeffs,"hf coeffs")
        mo_symmetry = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, hf.mo_coeff)
        print(mo_symmetry)
            
        mat = overlap_matrix_mo(hf_coeffs,overlap)
        print(mat,"mat")

        ov = overlap_mo(hf_coeffs[:,0],lowdin_coeffs[:,0],overlap)
        print(ov,"hf1")
        ov = overlap_mo(hf_coeffs[:,0],lowdin_coeffs[:,1],overlap)
        print(ov,"hf1")
        ov = overlap_mo(hf_coeffs[:,0],lowdin_coeffs[:,2],overlap)
        print(ov,"hf1")
        ov = overlap_mo(hf_coeffs[:,0],lowdin_coeffs[:,3],overlap)
        print(ov,"hf1")

        ov = overlap_mo(hf_coeffs[:,1],lowdin_coeffs[:,0],overlap)
        print(ov,"hf2")
        ov = overlap_mo(hf_coeffs[:,1],lowdin_coeffs[:,1],overlap)
        print(ov,"hf2")
        ov = overlap_mo(hf_coeffs[:,1],lowdin_coeffs[:,2],overlap)
        print(ov,"hf2")
        ov = overlap_mo(hf_coeffs[:,1],lowdin_coeffs[:,3],overlap)
        print(ov,"hf2")
        
        ov = overlap_mo(hf_coeffs[:,2],lowdin_coeffs[:,0],overlap)
        print(ov,"hf3")
        ov = overlap_mo(hf_coeffs[:,2],lowdin_coeffs[:,1],overlap)
        print(ov,"hf3")
        ov = overlap_mo(hf_coeffs[:,2],lowdin_coeffs[:,2],overlap)
        print(ov,"hf3")
        ov = overlap_mo(hf_coeffs[:,2],lowdin_coeffs[:,3],overlap)
        print(ov,"hf3")



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

        # get the states semi localised (core and csf)
        sloc_states_dets,sloc_states_coeffs = get_sloc_func(hf_coeffs,lowdin_coeffs,overlap,sloc_grouped)


        dets_list_matrix = RHF_dets_list + list_CSF_dets + sloc_states_dets 
        coeffs_list_matrix = RHF_coeffs_list + list_CSF_coeffs + sloc_states_coeffs 
        H_matrix = get_H_CSF_matrix(dets_list_matrix,coeffs_list_matrix,h_lowdin,g_lowdin,mol.energy_nuc())
        S = overlap_matrix_CSF(dets_list_matrix,coeffs_list_matrix)
        S_inv = np.linalg.inv(S)
        S_inv_H_matrix = np.matmul(S_inv,H_matrix)
        eigenvalues,eigenvectors = np.linalg.eig(S_inv_H_matrix)
        eigenvalue_min = 1000
        for eigenvalue,eigenvector in zip(eigenvalues,eigenvectors.T):
            if eigenvalue < eigenvalue_min:
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
            #plt.plot(r_array,RHF_energy_list)
            #plt.show()
        return RHF_energy_list[1:]
    else:
        if savetxt !=None:
            np.savetxt(f"{savetxt}.data",return_values_matrix)
            np.savetxt(f"{savetxt}_coeffs.data",return_coeffs_matrix)
            #plt.plot(r_array,RHF_energy_list)
            #plt.show()
        return RHF_energy_list

r = 1.2
mol = H4_mol(r)


overlap = mol.intor('int1e_ovlp')
lowdin_coeffs = get_symmetric_mo_coeffs(overlap)
csf002 = CSF(mol, lowdin_coeffs, 0.0, [], [0, 1, 2, 3], active_space=(4, 4), csf_build="genealogical", g_coupling="+-+-")
csf001 = CSF(mol,lowdin_coeffs,0.0,[],[0,1,2,3], active_space=(4,4),csf_build="genealogical",g_coupling="++--")
csf111 = CSF(mol,lowdin_coeffs,1.0,[],[0,1,2,3], active_space=(4,4),csf_build="genealogical",g_coupling="+++-")
csf001_dets,csf001_coeffs = csf001.dets,csf001.coeffs
csf111_dets,csf111_coeffs = csf111.dets,csf111.coeffs
csf_list_dets,csf_list_coeffs = csf002.dets,csf002.coeffs
#print(csf_list_dets,csf_list_coeffs,"after norm")
csf_list_dets_permute = permute_csf(csf_list_dets,[1,2,3,0])
#print(csf_list_dets_permute,"perm")
csf_list_dets, csf_list_coeffs = normalise_coeffs_det(csf_list_dets,csf_list_coeffs)
#print(csf_list_dets,csf_list_coeffs,"after norm")

csf_list = [csf_list_dets + csf_list_dets_permute] 
csf_coeffs = [csf_list_coeffs+csf_list_coeffs]

r_array = np.arange(0.5,1.5,0.05)
r_array = np.arange(0.5,3.55,0.05)
r_array = np.array([2.0])
#r_array = np.arange(0.5,0.6,0.05)

#e_nuc = []
#for r in r_array:
#    mol = H2_mol(r)
#    e_nuc.append(mol.energy_nuc())
#plt.plot(r_array,e_nuc)
#plt.show()


### get data for H4 ###

# separated states
x1 = get_Ecurve_CSF_RHF(r_array,[[1.0,1,0,1,0,1,0,1,0],[1.0,1,1,0,0,1,1,0,0]],[csf_list_dets,csf_list_dets_permute],[csf_list_coeffs,csf_list_coeffs],H4_mol,HF_coeffs=np.array([[1,1,1,1],[1,1,-1,-1],[1,-1.0,-1.0,1.0],[1.0,-1.0,1.0,-1.0]]),number_sloc_states=4,sloc_grouped=False,get_sloc_func=get_sloc_H4,RHF_states_grouped=False,savetxt="H4_curve")

# with different HF
#x2 = get_Ecurve_CSF_RHF(r_array,[[1.0,1,0,1,0,1,0,1,0],[1.0,1,1,0,0,1,1,0,0]],[csf_list_dets,csf_list_dets_permute],[csf_list_coeffs,csf_list_coeffs],H4_mol,HF_coeffs=np.array([[1,0,1,1],[1,1,0,-1],[1,0,-1,1],[1,-1,0,-1]]),number_sloc_states=4,sloc_grouped=False,get_sloc_func=get_sloc_H4,RHF_states_grouped=False,savetxt="H4_curve_hf")
#x3 = get_Ecurve_CSF_RHF(r_array,[[1.0,1,0,1,0,1,0,1,0],[1.0,1,1,0,0,1,1,0,0]],[csf_list_dets,csf_list_dets_permute],[csf_list_coeffs,csf_list_coeffs],H4_mol,HF_coeffs=np.array([[1,1,0,1],[1,0,1,-1],[1,-1,0,1],[1,0,-1,-1]]),number_sloc_states=4,sloc_grouped=False,get_sloc_func=get_sloc_H4,RHF_states_grouped=False,savetxt="H4_curve_permute")
#x = get_Ecurve_CSF_RHF(r_array,[[1.0,1,0,1,0,1,0,1,0],[1.0,1,1,0,0,1,1,0,0]],[csf_list_dets,csf_list_dets_permute],[csf_list_coeffs,csf_list_coeffs],H4_mol,HF_coeffs=None,number_sloc_states=4,sloc_grouped=False,get_sloc_func=get_sloc_H4,RHF_states_grouped=False,savetxt="H4_curve_nohf")
exit()
# with other CSFs
#x = get_Ecurve_CSF_RHF(r_array,[[1.0,1,0,1,0,1,0,1,0],[1.0,1,1,0,0,1,1,0,0]],[csf_list_dets,csf_list_dets_permute,csf001_dets,csf111_dets],[csf_list_coeffs,csf_list_coeffs,csf001_coeffs,csf111_coeffs],H4_mol,HF_coeffs=np.array([[1,1,1,1],[1,1,-1,-1],[1,-1,-1,1],[1,-1,1,-1]]),number_sloc_states=4,sloc_grouped=False,get_sloc_func=get_sloc_H4,RHF_states_grouped=False,savetxt="H4_curve")

# grouped states
#x = get_Ecurve_CSF_RHF(r_array,[[1.0,1,0,1,0,1,0,1,0],[1.0,1,1,0,0,1,1,0,0]],csf_list,csf_coeffs,H4_mol,HF_coeffs=np.array([[1,1,1,1],[1,1,-1,-1],[1,-1,-1,1],[1,-1,1,-1]]),number_sloc_states=1,sloc_grouped=True,get_sloc_func=get_sloc_H4,RHF_states_grouped=True,savetxt="H4_curve")
#x = get_Ecurve_CSF_RHF(r_array,[[1.0,1,0,1,0,1,0,1,0],[1.0,1,1,0,0,1,1,0,0]],[],[],H4_mol,RHF_states_grouped=False,savetxt="H4_curve")
#x = get_Ecurve_CSF_RHF(r_array,[],[],[],H4_mol,RHF_states_grouped=False,savetxt="H4_curve")
#x = get_Ecurve_CSF_RHF(r_array,[],[csf_list_dets,csf_list_dets_permute],[csf_list_coeffs,csf_list_coeffs],H4_mol,RHF_states_grouped=False,savetxt="H4_curve")

#x = get_Ecurve_CSF_RHF(r_array,[[1.0,1,0,1,0,1,0,1,0],[1.0,1,1,0,0,1,1,0,0]],csf_list,csf_coeffs,H4_mol,RHF_states_grouped=True,bond_length=None,savetxt="H4_curve")



# get data for H2
#x = get_Ecurve_CSF_RHF(r_array,[[1.0,1,0,1,0]],[[[1.0,1,0,0,1],[1.0,0,1,1,0]]],[[1/np.sqrt(2),1/np.sqrt(2)]],H2_mol,bond_length=None,savetxt="H2_curve")

r = 1
mol = H6_mol(r)
hf = np.array([[1,1,1,1,1,1],[2,1,-1,-2,-1,1],[0,3/2,3/2,0,-3/2,-3/2],[2,-1,-1,2,-1,-1],[0,3/2,-3/2,0,3/2,-3/2],[1,-1,1,-1,1,-1]])
hf = hf.T
overlap = mol.intor('int1e_ovlp')
lowdin_coeffs = get_symmetric_mo_coeffs(overlap)
csf001 = CSF(mol,lowdin_coeffs,0.0,[],[0,1,2,3,4,5],active_space=(6,6),csf_build="genealogical",g_coupling="+-+-+-")
csf_list_dets,csf_list_coeffs = csf001.dets,csf002.coeffs
# get data for H6
x = get_Ecurve_CSF_RHF(r_array,[[1.0,1,1,1,0,0,0,1,1,1,0,0,0]],[csf_list_dets],[csf_list_coeffs],H6_mol,HF_coeffs=hf,number_sloc_states=0,sloc_grouped=False,get_sloc_func=get_sloc_empty,RHF_states_grouped=False,savetxt="H6_curve")
exit()

