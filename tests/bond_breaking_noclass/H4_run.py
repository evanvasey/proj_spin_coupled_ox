import numpy as np
import matplotlib.pyplot as plt
from Hamiltonian import get_Ecurve_CSF_RHF
from molecules import H4_mol
from Lowdin_Orthogonalisation import get_symmetric_mo_coeffs
from sloc_states import get_sloc_H4,get_sloc_empty
from plot import plot_E_curve,plot_ci_curve 
from FCI import run_fci_over_r_array_and_save
from CSF import CSF
from CSF_tools import permute_csf,normalise_coeffs_det


# get CSFs
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

csf_list_dets_permute = permute_csf(csf_list_dets,[1,2,3,0])

#csf_list_dets, csf_list_coeffs = normalise_coeffs_det(csf_list_dets,csf_list_coeffs)

print(csf_list_dets)
print(csf_list_coeffs)
print(csf_list_dets_permute)
csf_list = [csf_list_dets + csf_list_dets_permute] 
csf_coeffs = [csf_list_coeffs+csf_list_coeffs]



### produce E curve and coeffs for H4 ###

# separated states

r_array = np.arange(0.5,3.55,0.05)

RHF_dets = [[1.0,1,0,1,0,1,0,1,0],[1.0,1,1,0,0,1,1,0,0]]
CSF_dets = [csf_list_dets,csf_list_dets_permute]
CSF_coeffs = [csf_list_coeffs,csf_list_coeffs]
CSF_dets = []
CSF_coeffs = []
CSF_dets = [csf_list_dets + csf_list_dets_permute]
CSF_coeffs = [csf002.coeffs + csf002.coeffs]
csf_list_dets = [[1.0,1,0,1,0,0,1,0,1],[1.0,1,0,0,1,0,1,1,0],[1.0,0,1,1,0,1,0,0,1],[1.0,0,1,0,1,1,0,1,0]]
csf_list_coeffs = [-1/2,-1/2,-1/2,-1/2]
csf_list_dets_permute = [[1.0,0,1,0,1,1,0,1,0],[1.0,1,1,0,0,0,0,1,1],[1.0,0,0,1,1,1,1,0,0],[1.0,1,0,1,0,0,1,0,1]]
csf_list_coeffs_permute = [1/2,1/2,1/2,1/2]
CSF_dets = [csf_list_dets,csf_list_dets_permute]
CSF_coeffs = [csf_list_coeffs,csf_list_coeffs_permute]




hf_coeffs = np.array([[1,1,0,1],[1,0,1,-1],[1,-1.0,0,1.0],[1,0,-1,-1]])
hf_coeffs = np.array([[1,1,1,1],[1.5,0.5,-1.5,-0.5],[0.5,1.5,-0.5,-1.5],[1,-1,1,-1]])
hf_coeffs = hf_coeffs.T
hf_coeffs = np.array([[1,1,1,1],[1,1,-1,-1],[1,-1,-1,1],[1,-1,1,-1]])

get_Ecurve_CSF_RHF(r_array,RHF_dets,CSF_dets,CSF_coeffs,H4_mol,HF_coeffs=hf_coeffs,number_sloc_states=4,sloc_grouped=False,get_sloc_func=get_sloc_H4,RHF_states_grouped=False,savetxt="data/H4_curve")
#get_Ecurve_CSF_RHF(r_array,RHF_dets,[],[],H4_mol,HF_coeffs=hf_coeffs,number_sloc_states=0,sloc_grouped=False,get_sloc_func=get_sloc_empty,RHF_states_grouped=False,savetxt="data/H4_curve_rhf3")
#get_Ecurve_CSF_RHF(r_array,[],CSF_dets,CSF_coeffs,H4_mol,HF_coeffs=hf_coeffs,number_sloc_states=0,sloc_grouped=False,get_sloc_func=get_sloc_empty,RHF_states_grouped=False,savetxt="data/H4_curve_csf")
#get_Ecurve_CSF_RHF(r_array,[],[],[],H4_mol,HF_coeffs=hf_coeffs,number_sloc_states=4,sloc_grouped=False,get_sloc_func=get_sloc_H4,RHF_states_grouped=False,savetxt="data/H4_curve_sloc")
#get_Ecurve_CSF_RHF(r_array,RHF_dets,CSF_dets,CSF_coeffs,H4_mol,HF_coeffs=hf_coeffs,number_sloc_states=1,sloc_grouped=True,get_sloc_func=get_sloc_H4,RHF_states_grouped=True,savetxt="data/H4_curve_grp")



# separated states different hf coeffs

r_array = np.arange(0.5,3.55,0.05)

RHF_dets = [[1.0,1,0,1,0,1,0,1,0],[1.0,1,1,0,0,1,1,0,0]]
CSF_dets = [csf_list_dets,csf_list_dets_permute]
CSF_coeffs = [csf_list_coeffs,csf_list_coeffs]

hf_coeffs = np.array([[1,1,0,1],[1,0,1,-1],[1,-1.0,0,1.0],[1.0,0,-1.0,-1.0]])

#get_Ecurve_CSF_RHF(r_array,RHF_dets,CSF_dets,CSF_coeffs,H4_mol,HF_coeffs=hf_coeffs,number_sloc_states=4,sloc_grouped=False,get_sloc_func=get_sloc_H4,RHF_states_grouped=False,savetxt="data/H4_curvehf")



# separated states hf coeffs from pyscf

r_array = np.arange(0.5,3.55,0.05)

RHF_dets = [[1.0,1,0,1,0,1,0,1,0],[1.0,1,1,0,0,1,1,0,0]]
CSF_dets = [csf_list_dets,csf_list_dets_permute]
CSF_coeffs = [csf_list_coeffs,csf_list_coeffs]

hf_coeffs = np.array([[1,1,0,1],[1,0,1,-1],[1,-1.0,0,1.0],[1.0,0,-1.0,-1.0]])

#get_Ecurve_CSF_RHF(r_array,RHF_dets,CSF_dets,CSF_coeffs,H4_mol,HF_coeffs=hf_coeffs,number_sloc_states=4,sloc_grouped=False,get_sloc_func=get_sloc_H4,RHF_states_grouped=False,savetxt="data/H4_curvepyscf")


# grouped states


r_array = np.arange(0.5,4.0,0.5)
r_array = np.array([3.0,3.5])


hf_coeffs = np.array([[1,1,1,1],[1,1,-1,-1],[1,-1,-1,1],[1,-1,1,-1]])
print("RHF ONLY")
#get_Ecurve_CSF_RHF(r_array,RHF_dets,[],[],H4_mol,HF_coeffs=hf_coeffs,number_sloc_states=0,sloc_grouped=False,get_sloc_func=get_sloc_empty,RHF_states_grouped=False,savetxt="data/H4_curve_rhf")


CSF_dets = [csf_list_dets+csf_list_dets_permute]
CSF_coeffs = [csf_list_coeffs+csf_list_coeffs]

CSF_dets1,CSF_coeffs1 = normalise_coeffs_det(CSF_dets[0],CSF_coeffs[0]) 
CSF_dets = [CSF_dets1]
CSF_coeffs = [CSF_coeffs1]

print("GROUPED STATES")
#get_Ecurve_CSF_RHF(r_array,RHF_dets,CSF_dets,CSF_coeffs,H4_mol,HF_coeffs=hf_coeffs,number_sloc_states=1,sloc_grouped=True,get_sloc_func=get_sloc_H4,RHF_states_grouped=True,savetxt="data/H4_curve_grp")





### E curve coeffs END ###





### FCI ###

#fci_energies,mcscf_energies = run_fci_over_r_array_and_save(r_array, H4_mol,4,4,savetxt="data/H4_FCI")
#plt.plot(r_array,fci_energies)
#plt.plot(r_array,mcscf_energies)
#plt.show()

### FCI END ###


### PLOT ###

plot_E_curve("data/H4_curve.data","plots/H4_curve.pdf",[r"$\Phi_{LC}$",r"$\Phi_{RHF1}$",r"$\Phi_{RHF1}$",r"$\Phi_2$",r"$\Phi_2$(permute)",r"$\sigma_L$",r"$\sigma_R$",r"$\sigma_T$",r"$\sigma_B$"],(0.7,3.5),(-2.2,-1.0),n_H=4,true_wavefunction="data/H4_FCI.data",true_wavefunction_index=1)
plot_ci_curve("data/H4_curve_coeffs.data","plots/H4_curve_coeffs.pdf",[r"$\Phi_{RHF1}$",r"$\Phi_{RHF2}$",r"$\Phi_2$",r"$\Phi_2$(permute)",r"$\sigma_L$",r"$\sigma_R$",r"$\sigma_T$",r"$\sigma_B$"])


plot_E_curve("data/H4_curve_grp.data","plots/H4_curve_grp.pdf",[r"$\Phi_{LC}$",r"$\Phi_{RHF1}$",r"$\Phi_{RHF1}$",r"$\Phi_2$",r"$\Phi_2$(permute)",r"$\sigma_L$",r"$\sigma_R$",r"$\sigma_T$",r"$\sigma_B$"],(0.7,3.5),(-2.2,-1.0),n_H=4,true_wavefunction="data/H4_FCI.data",true_wavefunction_index=1)
plot_ci_curve("data/H4_curve_grp_coeffs.data","plots/H4_curve_grp_coeffs.pdf",[r"$\Phi_{RHF1}$",r"$\Phi_{RHF2}$",r"$\Phi_2$",r"$\Phi_2$(permute)",r"$\sigma_L$",r"$\sigma_R$",r"$\sigma_T$",r"$\sigma_B$"])


plot_E_curve("data/H4_curvepyscf.data","plots/H4_curvepyscf.pdf",[r"$\Phi_{LC}$",r"$\Phi_{RHF1}$",r"$\Phi_{RHF1}$",r"$\Phi_2$",r"$\Phi_2$(permute)",r"$\sigma_L$",r"$\sigma_R$",r"$\sigma_T$",r"$\sigma_B$"],(0.7,3.5),(-2.2,-1.0),n_H=4,true_wavefunction="data/H4_FCI.data",true_wavefunction_index=1)
plot_ci_curve("data/H4_curvepyscf_coeffs.data","plots/H4_curvepyscf_coeffs.pdf",[r"$\Phi_{RHF1}$",r"$\Phi_{RHF2}$",r"$\Phi_2$",r"$\Phi_2$(permute)",r"$\sigma_L$",r"$\sigma_R$",r"$\sigma_T$",r"$\sigma_B$"])


### PLOT END ###
