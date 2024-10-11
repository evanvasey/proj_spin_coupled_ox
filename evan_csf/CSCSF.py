r"""
Closed shell CSF.
"""

import numpy as np
from pyscf import gto
from GenericCSF import GenericCSF
from Operators.BasicOperators import create


class CSCSF(GenericCSF):
    def __init__(self, mol: gto.Mole, stot: float, n_core: int, n_act=0):
        # By default, number of active orbitals n_act = 0 for closed shell CSF
        self.mol = mol
        super().__init__(stot, mol.spin / 2, n_core, n_act, mol.nao - n_core - n_act, mol.nelectron)
        self.dets_sq, self.csf_coeffs = self.get_cscsf()

    def get_cscsf(self):
        alpha_idxs = np.arange(self.n_alpha - self.n_core)
        beta_idxs = np.arange(self.n_act, self.n_beta - self.n_core)

        ket = [0] * (self.n_act * 2 + 1)
        ket[0] = 1
        for _, i in enumerate(beta_idxs):
            create(i + 1 + self.n_act, ket)
        for _, i in enumerate(alpha_idxs):
            create(i + 1, ket)
        return [ket], [[1.]]