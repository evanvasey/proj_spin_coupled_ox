import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci, mcscf  


def run_fci_over_r_array_and_save(r_array, mol_func, ncas,nelecas,savetxt):
    fci_energies = []
    mcscf_energies = []

    for r in r_array:
        print(f"r = {r}")
        # Create the molecule object for the current bond length r
        mol = mol_func(r)
        
        
        # Perform a Restricted Hartree-Fock (RHF) calculation
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10 
        mf.kernel()
        print(mf.mo_coeff)
        
        mc = mcscf.CASSCF(mf,ncas,nelecas)
        mc.kernel()
        
        # Perform Full Configuration Interaction (FCI) calculation
        cisolver = fci.FCI(mf)
        e_fci, ci_vector = cisolver.kernel()


        # Store the FCI energy
        fci_energies.append(e_fci)
        mcscf_energies.append(mc.e_tot)

    # Save r_array and fci_energies to a file

    data = np.column_stack((r_array, fci_energies, mcscf_energies))
    np.savetxt(savetxt+".data", data)
    return fci_energies,mcscf_energies
"""
# Example usage:
r_array = np.arange(0.5, 3.5, 0.01)  # Define a range of bond lengths


fci_energies,mcscf_energies = run_fci_over_r_array_and_save(r_array, H2_mol,2,2,savetxt="data/H2_FCI")
plt.plot(r_array,fci_energies)
plt.plot(r_array,mcscf_energies)
plt.show()
exit()
r_array = np.arange(0.5, 3.5, 0.05)  # Define a range of bond lengths




#fci_energies = run_fci_over_r_array_and_save(r_array, H6_mol, savetxt="H6_FCI")
fci_energies,mcscf_energies = run_fci_over_r_array_and_save(r_array, H6_mol,6,6,savetxt="data/H6_FCI")
plt.plot(r_array,fci_energies)
plt.plot(r_array,mcscf_energies)
plt.show()

fci_energies,mcscf_energies = run_fci_over_r_array_and_save(r_array, H4_mol,4,4,savetxt="data/H4_FCI")
plt.plot(r_array,fci_energies)
plt.plot(r_array,mcscf_energies)
plt.show()
"""
r_array = np.arange(-2,2,0.1)
fci_energies,mcscf_energies = run_fci_over_r_array_and_save(r_array, H2_H_mol,3,3, savetxt="data/H2H_FCI")
plt.plot(r_array,fci_energies)
plt.plot(r_array,mcscf_energies)
plt.show()
