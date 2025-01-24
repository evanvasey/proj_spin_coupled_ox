import numpy as np
import matplotlib.pyplot as plt
from Hamiltonian import get_Ecurve_CSF_RHF
from molecules import H6_mol
from Lowdin_Orthogonalisation import get_symmetric_mo_coeffs
from sloc_states import get_sloc_empty
from plot import plot_E_curve,plot_ci_curve 
from FCI import run_fci_over_r_array_and_save
from CSF import CSF
from CSF_tools import permute_csf,normalise_coeffs_det,overlap_CSF



r = 1
r_array = np.arange(0.5, 3.5, 0.1)  # Define a range of bond lengths
mol = H6_mol(r)
overlap = mol.intor('int1e_ovlp')
lowdin_coeffs = get_symmetric_mo_coeffs(overlap)
csf001 = CSF(mol,lowdin_coeffs,0.0,[],[0,1,2,3,4,5],active_space=(6,6),csf_build="genealogical",g_coupling="+-+-+-")
csf_list_dets,csf_list_coeffs = csf001.dets,csf001.coeffs
csf_list_dets_permute = permute_csf(csf_list_dets,[1,2,3,4,5,0])
csf_list_dets_permute2 =permute_csf(csf_list_dets,[2,3,4,5,0,1])
csf_list_dets_permute3 =permute_csf(csf_list_dets,[3,4,5,0,1,2])

print(overlap_CSF(csf_list_dets,csf_list_coeffs,csf_list_dets_permute3,csf_list_coeffs))
print(csf_list_dets)
print(csf_list_coeffs)
print(csf_list_dets_permute)
print(csf_list_dets_permute2)
### produce E curve and coeffs for H6 ###

RHF_dets = [[1.0,1,1,1,0,0,0,1,1,1,0,0,0]]
CSF_dets = [csf_list_dets,csf_list_dets_permute]
CSF_coeffs = [csf_list_coeffs,csf_list_coeffs]

hf = np.array([[1,1,1,1,1,1],[2,1,-1,-2,-1,1],[0,3/2,3/2,0,-3/2,-3/2],[2,-1,-1,2,-1,-1],[0,3/2,-3/2,0,3/2,-3/2],[1,-1,1,-1,1,-1]])
hf = np.array([[1,1,1,1,1,1],[1,1,-1,-1,-1,1],[0,1,1,0,-1,-1],[1,-1,-1,1,-1,-1],[0,1,-1,0,1,-1],[1,-1,1,-1,1,-1]])
hf = hf.T

#x = get_Ecurve_CSF_RHF(r_array,RHF_dets,CSF_dets,CSF_coeffs,H6_mol,HF_coeffs=hf,number_sloc_states=0,sloc_grouped=False,get_sloc_func=get_sloc_empty,RHF_states_grouped=False,savetxt="data/H6_curve")


### E curve coeffs END ###


hf = np.array([[1,1,1,1,1,1],[2,1,-1,-2,-1,1],[0,3/2,3/2,0,-3/2,-3/2],[2,-1,-1,2,-1,-1],[0,3/2,-3/2,0,3/2,-3/2],[1,-1,1,-1,1,-1]])
hf = hf.T
#x = get_Ecurve_CSF_RHF(r_array,RHF_dets,CSF_dets,CSF_coeffs,H6_mol,HF_coeffs=hf,number_sloc_states=0,sloc_grouped=False,get_sloc_func=get_sloc_empty,RHF_states_grouped=False,savetxt="data/H6_curve_hf")



### FCI ###

#fci_energies,mcscf_energies = run_fci_over_r_array_and_save(r_array, H6_mol,6,6,savetxt="data/H6_FCI")
#plt.plot(r_array,fci_energies)
#plt.plot(r_array,mcscf_energies)
#plt.show()

### FCI END ###


### PLOT ###


plot_E_curve("data/H6_curve_hf.data","plots/H6_curve_hf.pdf",[r"$\Phi_{LC}$",r"$\Phi_{RHF1}$",r"$\Phi_{RHF1}$",r"$\Phi_2$",r"$\Phi_2$(permute)",r"$\sigma_L$",r"$\sigma_R$",r"$\sigma_T$",r"$\sigma_B$"],(0.7,3.5),(-3.4,-1.0),n_H=6,true_wavefunction="data/H6_FCI.data",true_wavefunction_index=1)
plot_ci_curve("data/H6_curve_hf_coeffs.data","plots/H6_curve_hf_coeffs.pdf",[r"$\Phi_{RHF1}$",r"$\Phi_{RHF2}$",r"$\Phi_2$",r"$\Phi_2$(permute)",r"$\sigma_L$",r"$\sigma_R$",r"$\sigma_T$",r"$\sigma_B$"])


### PLOT END ###

