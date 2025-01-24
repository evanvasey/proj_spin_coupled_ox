import numpy as np
import matplotlib.pyplot as plt
from state import mc_state
from molecules import H4_mol
from lowdin_orthogonalisation import get_symmetric_mo_coeffs
from sloc_states import get_sloc_H4,get_sloc_empty
from bond_breaking import get_Ecurve_CSF_RHF
from plot import plot_E_curve,plot_ci_curve 
from fci import run_fci_over_r_array_and_save











### CSFs ###0

csf_list_dets = [[1.0,1,0,1,0,0,1,0,1],[1.0,1,0,0,1,0,1,1,0],[1.0,0,1,1,0,1,0,0,1],[1.0,0,1,0,1,1,0,1,0]]
csf_list_coeffs = [-1/2,-1/2,-1/2,-1/2]
csf_list_dets_permute = [[1.0,0,1,0,1,1,0,1,0],[1.0,1,1,0,0,0,0,1,1],[1.0,0,0,1,1,1,1,0,0],[1.0,1,0,1,0,0,1,0,1]]
csf_list_coeffs_permute = [1/2,1/2,1/2,1/2]

CSF_state1 = mc_state(csf_list_dets,csf_list_coeffs)
CSF_state2 = mc_state(csf_list_dets_permute,csf_list_coeffs_permute)

CSF_states = [CSF_state1,CSF_state2]

### produce E curve and coeffs for H4 ###

r_array = np.arange(0.5,3.55,0.05)

RHF_dets = [[1.0,1,0,1,0,1,0,1,0],[1.0,1,1,0,0,1,1,0,0]]
RHF_state1 = mc_state([[1.0,1,1,0,0,1,1,0,0]])
RHF_state2 = mc_state([[1.0,1,0,1,0,1,0,1,0]])
RHF_states = [RHF_state2,RHF_state1]
hf_coeffs = np.array([[1,1,1,1],[1,1,-1,-1],[1,-1,-1,1],[1,-1,1,-1]])

# separated states
get_Ecurve_CSF_RHF(r_array,RHF_states,CSF_states,H4_mol,HF_coeffs=hf_coeffs,number_sloc_states=4,sloc_grouped=False,get_sloc_func=get_sloc_H4,savetxt="data/H4_curve")

# only RHF

#get_Ecurve_CSF_RHF(r_array,RHF_states,[],H4_mol,HF_coeffs=hf_coeffs,number_sloc_states=0,sloc_grouped=False,get_sloc_func=get_sloc_empty,savetxt="data/H4_curve_onlyrhf")



### E curve coeffs END ###





### FCI ###

fci_energies,mcscf_energies = run_fci_over_r_array_and_save(r_array, H4_mol,4,4,savetxt="data/H4_FCI")
#plt.plot(r_array,fci_energies)
#plt.plot(r_array,mcscf_energies)
#plt.show()

### FCI END ###


### PLOT ###

plot_E_curve("data/H4_curve.data","plots/H4_curve.pdf",[r"$\Phi_{LC}$",r"$\Phi_{RHF1}$",r"$\Phi_{RHF1}$",r"$\Phi_2$",r"$\Phi_2$(permute)",r"$\sigma_L$",r"$\sigma_R$",r"$\sigma_T$",r"$\sigma_B$"],(0.7,3.5),(-2.2,-1.0),n_H=4,true_wavefunction="data/H4_FCI.data",true_wavefunction_index=1)
plot_ci_curve("data/H4_curve_coeffs.data","plots/H4_curve_coeffs.pdf",[r"$\Phi_{RHF1}$",r"$\Phi_{RHF2}$",r"$\Phi_2$",r"$\Phi_2$(permute)",r"$\sigma_L$",r"$\sigma_R$",r"$\sigma_T$",r"$\sigma_B$"])

### PLOT END ###
