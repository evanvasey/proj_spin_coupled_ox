import numpy as np
import matplotlib.pyplot as plt
from state import mc_state
from bond_breaking import get_Ecurve_CSF_RHF
from molecules import H2_mol
from lowdin_orthogonalisation import get_symmetric_mo_coeffs
from sloc_states import get_sloc_empty
from plot import plot_E_curve,plot_ci_curve 
from fci import run_fci_over_r_array_and_save




### produce E curve and coeffs for H2 ###

r_array = np.arange(0.5,3.55,0.05)

RHF_dets = [[1.0,1,0,1,0]]
RHF_states = [mc_state(RHF_dets)]

CSF_dets = [[1.0,1,0,0,1],[1.0,0,1,1,0]]
CSF_coeffs = [1/np.sqrt(2),1/np.sqrt(2)]
CSF_states = [mc_state(CSF_dets,CSF_coeffs)]

hf_coeffs = np.array([[1,1],[1,-1]]) 


get_Ecurve_CSF_RHF(r_array,RHF_states,CSF_states,H2_mol,HF_coeffs=hf_coeffs,number_sloc_states=0,sloc_grouped=False,get_sloc_func=get_sloc_empty,bond_length=None,savetxt="data/H2_curve")

### E curve coeffs END ###





### FCI ###

r_array = np.arange(0.5, 3.55, 0.05)  
fci_energies,mcscf_energies = run_fci_over_r_array_and_save(r_array, H2_mol,2,2,savetxt="data/H2_FCI")
#plt.plot(r_array,fci_energies)
#plt.plot(r_array,mcscf_energies)
#plt.show()
#plt.clf()

### FCI END ###






### PLOT ###

plot_E_curve("data/H2_curve.data","plots/H2_curve.pdf",[r"$\Phi_{LC}$",r"$\Phi_{RHF}$",r"$\Phi_2$"],(0.5,3.5),(-1.15,-0.6),n_H=2,true_wavefunction="data/H2_FCI.data",true_wavefunction_index=1)
plot_ci_curve("data/H2_curve_coeffs.data","plots/H2_curve_coeffs.pdf",[r"$\Phi_{RHF}$",r"$\Phi_2$"])

### PLOT END ###
