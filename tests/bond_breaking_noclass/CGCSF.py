r"""
Construct CSFs via Clebsch-Gordon coupling
"""
import numpy as np
from pyscf import gto
from typing import Tuple

from ActiveCSF import ActiveCSF
from CSFTools.ClebschGordon import get_general_tensorprod, get_local_g_coupling, get_cg, take_csf_tensorprod
from CSFTools.PermutationTools import get_phase_factor
from GenericCSF import GenericCSF


class CGCSF(GenericCSF):
    def __init__(self, mol: gto.Mole, stot: float, j1: float, j2: float,
                 cas1: Tuple[int, int], cas2: Tuple[int, int],
                 n_core: int, n_act: int):
        super().__init__(stot, mol.spin // 2, n_core, n_act, mol.nao - n_core - n_act, mol.nelectron)
        self.mol = mol
        self.dets_sq, self.csf_coeffs = self.get_cgcsf(j1, j2, cas1, cas2, self.stot, self.mol.spin // 2)

    def construct_csf_det(self, s, ms, cas, g_coupling):
        r"""
        """
        csfrep = ActiveCSF(s, ms, cas, g_coupling)
        return csfrep.dets_sq, csfrep.csf_coeffs

    def get_coupled_csf(self, j1, j2, cas1, cas2, j, m):
        # We want to form the tensor product of CSFs
        states_required = get_general_tensorprod(j1, j2, j, m)
        kets = []
        coeffs = []
        for _, state in enumerate(states_required):
            m1 = state[1]
            m2 = state[3]
            g_a = get_local_g_coupling(cas1[0], j1)
            kets_a, coeffs_a = self.construct_csf_det(j1, m1, cas1, g_a)
            g_b = get_local_g_coupling(cas2[0], j2)
            kets_b, coeffs_b = self.construct_csf_det(j2, m2, cas2, g_b)
            cg = get_cg(j1, j2, j, m1, m2, m1 + m2)
            kets_total, coeffs_total = take_csf_tensorprod(kets_a, coeffs_a, kets_b, coeffs_b, cg)
            kets.extend(kets_total)
            coeffs.extend(coeffs_total)
        return kets, coeffs

    def get_cgcsf(self, j1, j2, cas1, cas2, j, m):
        kets, coeffs = self.get_coupled_csf(j1, j2, cas1, cas2, j, m)
        # We have to process the kets, however
        newkets = []
        for _, ket in enumerate(kets):
            n = (len(ket) - 1) // 2
            ket[0] = get_phase_factor(list(np.nonzero(ket[1:n + 1])[0]), list(np.nonzero(ket[n + 1:])[0]))
            newkets.append(ket)
        return newkets, coeffs