import numpy as np
import matplotlib.pyplot as plt
from Hamiltonian import get_Ecurve_CSF_RHF
from molecules import H6_mol
from Lowdin_Orthogonalisation import get_symmetric_mo_coeffs
from sloc_states import get_sloc_empty
from plot import plot_E_curve,plot_ci_curve 
from FCI import run_fci_over_r_array_and_save
from CSF import CSF
from CSF_tools import permute_csf,normalise_coeffs_det



r = 1
r_array = np.arange(0.5, 3.5, 0.05)  # Define a range of bond lengths
mol = H6_mol(r)
hf = np.array([[1,1,1,1,1,1],[2,1,-1,-2,-1,1],[0,3/2,3/2,0,-3/2,-3/2],[2,-1,-1,2,-1,-1],[0,3/2,-3/2,0,3/2,-3/2],[1,-1,1,-1,1,-1]])
hf = hf.T
overlap = mol.intor('int1e_ovlp')
lowdin_coeffs = get_symmetric_mo_coeffs(overlap)
csf001 = CSF(mol,lowdin_coeffs,0.0,[],[0,1,2,3,4,5],active_space=(6,6),csf_build="genealogical",g_coupling="+-+-+-")
csf_list_dets,csf_list_coeffs = csf001.dets,csf001.coeffs
# get data for H6
x = get_Ecurve_CSF_RHF(r_array,[[1.0,1,1,1,0,0,0,1,1,1,0,0,0]],[csf_list_dets],[csf_list_coeffs],H6_mol,HF_coeffs=hf,number_sloc_states=0,sloc_grouped=False,get_sloc_func=get_sloc_empty,RHF_states_grouped=False,savetxt="H6_curve")



### FCI ###
r_array = np.arange(0.5, 3.5, 0.05)  # Define a range of bond lengths

#fci_energies = run_fci_over_r_array_and_save(r_array, H6_mol, savetxt="H6_FCI")
fci_energies,mcscf_energies = run_fci_over_r_array_and_save(r_array, H6_mol,6,6,savetxt="data/H6_FCI")
plt.plot(r_array,fci_energies)
plt.plot(r_array,mcscf_energies)
plt.show()
