"""This is a general CSF that is a linear combination of other CSFs"""

import numpy as np
from pyscf import gto
from typing import List

from GenericCSF import GenericCSF


class LCCSF(GenericCSF):
    def __init__(self, mol: gto.Mole, stot: float,
                 n_core: int, n_act: int, csfs: List[GenericCSF], rel_weights: List):
        super().__init__(stot, mol.spin // 2, n_core, n_act, mol.nao - n_core - n_act, mol.nelectron)
        self.mol = mol
        self.dets_sq, self.csf_coeffs = self.get_lccsf(csfs, rel_weights)

    def get_unique_kets_and_coeffs(self, csfs, rel_weights):
        kets = []
        coeffs = []
        for i, csf in enumerate(csfs):
            csf_kets, csf_coeffs = csf.csf_instance.get_relevant_dets(csf.csf_instance.dets_sq,
                                                                      csf.csf_instance.csf_coeffs)
            for csf_ket in csf_kets:
                kets.append(csf_ket)
            for csf_coeff in csf_coeffs:
                coeffs.append(csf_coeff * rel_weights[i])
        unique_kets = []
        unique_coeffs = []
        for i, ket in enumerate(kets):
            if ket in unique_kets:
                unique_idx = unique_kets.index(ket)
                unique_coeffs[unique_idx] += coeffs[i]
            else:
                unique_kets.append(ket)
                unique_coeffs.append(coeffs[i])
        return unique_kets, unique_coeffs

    def get_lccsf(self, csfs, rel_weights):
        r"""
        Get determinants in second quantised formalism and their coefficients for the linear combination.
        """
        kets, coeffs = self.get_unique_kets_and_coeffs(csfs, rel_weights)
        norm = np.sum([x ** 2 for x in coeffs])     # Find norm
        coeffs = [x / np.sqrt(norm) for x in coeffs]    # Normalise
        return kets, coeffs