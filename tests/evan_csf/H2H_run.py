import numpy as np
import matplotlib.pyplot as plt
from Hamiltonian import get_Ecurve_CSF_RHF
from molecules import H2H_mol
from Lowdin_Orthogonalisation import get_symmetric_mo_coeffs
from sloc_states import get_sloc_H4
from plot import plot_E_curve,plot_ci_curve 
from FCI import run_fci_over_r_array_and_save
from CSF import CSF
from CSF_tools import permute_csf,normalise_coeffs_det


















### FCI ###

r_array = np.arange(-2,2,0.1)
fci_energies,mcscf_energies = run_fci_over_r_array_and_save(r_array, H2H_mol,3,3, savetxt="data/H2H_FCI")
plt.plot(r_array,fci_energies)
plt.plot(r_array,mcscf_energies)
plt.show()


### FCI END ###
