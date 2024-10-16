import numpy as np
from Hamiltonian import get_Ecurve_CSF_RHF
from molecules import H2_mol
from Lowdin_Orthogonalisation import get_symmetric_mo_coeffs
from sloc_states import get_sloc_H4
from plot import plot_E_curve,plot_ci_curve 
from FCI import run_fci_over_r_array_and_save
from CSF import CSF
from CSF_tools import permute_csf,normalise_coeffs_det




r_array = np.arange(0.5,3.55,0.05)

# get data for H2
x = get_Ecurve_CSF_RHF(r_array,[[1.0,1,0,1,0]],[[[1.0,1,0,0,1],[1.0,0,1,1,0]]],[[1/np.sqrt(2),1/np.sqrt(2)]],H2_mol,bond_length=None,savetxt="H2_curve")



### FCI ###

r_array = np.arange(0.5, 3.5, 0.01)  # Define a range of bond lengths
fci_energies,mcscf_energies = run_fci_over_r_array_and_save(r_array, H2_mol,2,2,savetxt="data/H2_FCI")
plt.plot(r_array,fci_energies)
plt.plot(r_array,mcscf_energies)
plt.show()

### FCI END ###
