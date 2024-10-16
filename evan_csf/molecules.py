import numpy as np
from pyscf import gto





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

                                             
def H2H_mol(r,h2_bond_length=0.74):
    if r > 0:
        r1 = -h2_bond_length
        r2 =  h2_bond_length + r
    elif r==0:
        r1 =  h2_bond_length
        r2 =  h2_bond_length
    else:
        r1 =  -h2_bond_length + r
        r2 = h2_bond_length
    mol = gto.Mole()
    mol.build(
            atom = f'''H 0 0 0; H 0 0 {r1}; H 0 0 {r2}''',
            basis = 'sto-3g',
            symmetry= False,
            spin = 1,
            charge = 0)

    return mol

