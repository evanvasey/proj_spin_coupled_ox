#!/usr/bin/python3
# This is code for a CSF, which can be formed in a variety of ways.
import copy
import sys

import numpy as np
from typing import List, Optional
from CGCSF import CGCSF
from GCCSF import GCCSF
from CSCSF import CSCSF
from LCCSF import LCCSF


class CSF():
    def __init__(self, mol, mo_coeff: np.ndarray, stot, core: List[int],
                 act: Optional[List[int]], active_space=None, g_coupling: str = None,
                 permutation: List[int] = None, csf_build: str = 'genealogical',
                 localstots: List[float] = None, active_subspaces: List[int] = None, csfs = None):

        # CSF should have:
        # 1. gto.Mol object
        # 2. MO coefficients in spin-orbital basis (mo_coeff)
        # 3. Core/ Act (List of core and active orbitals, respectively)
        self.mol = mol
        self.mo_coeff = mo_coeff
        self.core = core
        self.act = act
        if csfs is not None:
            self.csfs = csfs
        self.active_space = active_space
        self.csf_build = csf_build

        # Setup CSF variables
        if csf_build.lower() == 'cscsf':
            self.setup_cscsf(stot)
        else:
            self.setup_csf(stot, active_space, g_coupling, permutation, csf_build,
                           localstots, active_subspaces)

    def initialise(self):
        self.ovlp = self.mol.intor('int1e_ovlp')
        self.nao = self.ovlp.shape[0]

        # Initialise orbitals
        if self.act is not None:
            self.nact = len(self.act)
            self.vir = list(set(np.arange(self.mol.nao)) - set(self.core + self.act))
        else:
            self.nact = 0
            self.vir = list(set(np.arange(self.mol.nao)) - set(self.core))
        self.ncore = len(self.core)
        self.nvir = len(self.vir)

        # These are for spin-orbitals. For use in CC codes
        self.c = self.core + [self.nao + x for _, x in enumerate(self.core)]
        if self.act is not None:
            self.a = self.act + [self.nao + x for _, x in enumerate(self.act)]
        else:
            self.a = []
        self.v = self.vir + [self.nao + x for _, x in enumerate(self.vir)]


    def setup_csf(self, stot: float, active_space: List[int],
                  g_coupling: str, permutation: List[int],
                  csf_build, localstots, active_subspaces):

        self.ncas = active_space[0]  # Number of active orbitals
        self.stot = stot  # Total S value

        self.g_coupling = g_coupling  # Genealogical coupling pattern
        self.permutation = permutation  # Permutation for spin coupling

        self.csf_build = csf_build  # Method for constructing CSFs
        self.localstots = localstots  # Local spins if we are doing Clebsch-Gordon coupling
        self.active_subspaces = active_subspaces  # Local active spaces if we are doing Clebsch-Gordon coupling

        if isinstance(active_space[1], (int, np.integer)):
            nelecb = (active_space[1] - self.mol.spin) // 2
            neleca = active_space[1] - nelecb
            self.nelecas = (neleca, nelecb)  # Tuple of number of active electrons
        else:
            self.nelecas = np.asarray((active_space[1][0], active_space[1][1])).astype(int)

        ncorelec = self.mol.nelectron - sum(self.nelecas)
        assert ncorelec % 2 == 0
        assert ncorelec >= 0

        # Build CSF
        if csf_build.lower() == 'genealogical':
            self.csf_instance = GCCSF(self.mol, self.stot, len(self.core), len(self.act),
                                      self.g_coupling)
        elif csf_build.lower() == 'clebschgordon':
            assert localstots is not None, "Local spin quantum numbers (localstots) undefined"
            assert active_subspaces is not None, "Active subspaces (active_subspaces) undefined"
            assert active_subspaces[0] + active_subspaces[2] == active_space[0], "Mismatched number of active orbitals"
            self.csf_instance = CGCSF(self.mol, self.stot, localstots[0], localstots[1],
                                      (active_subspaces[0], active_subspaces[1]),
                                      (active_subspaces[2], active_subspaces[3]),
                                      len(self.core), len(self.act))
        elif csf_build.lower() == 'spinensemble':
            assert len(self.csfs) == 1
            hs_csf = self.csfs[0]
            assert np.isclose(0.5 * hs_csf.mol.spin, hs_csf.stot, rtol=0, atol=1e-10) # Asserts that CSF is high-spin (S = Ms)

            csfs = []
            rel_weights = []
            n_csfs = int(self.stot * 2) + 1
            for i in range(n_csfs):
                rel_weights.append(1 / (2 * self.stot + 1))
                # Construct new CSF with different Ms
                hs_csf.mol.spin -= 2 * i
                new_csf = CSF(hs_csf.mol, hs_csf.mo_coeff, hs_csf.stot, hs_csf.core, hs_csf.act,
                              active_space=hs_csf.active_space, csf_build=hs_csf.csf_build)
                csfs.append(new_csf)
            self.csf_instance = LCCSF(self.mol, self.stot, len(self.core), len(self.act), csfs, rel_weights)

        else:
            import sys
            sys.exit("The requested CSF build is not supported, exiting.")

        # Get relevant determinants and coefficients
        self.dets, self.coeffs = self.csf_instance.get_relevant_dets(self.csf_instance.dets_sq, self.csf_instance.csf_coeffs)

    def setup_cscsf(self, stot: float):

        self.stot = stot  # Total S value
        self.csf_build = 'cscsf'  # Method for constructing CSFs

        ncorelec = self.mol.nelectron
        assert ncorelec % 2 == 0
        assert ncorelec >= 0
        self.nelecas = (0, 0)
        self.ncas = 0
        self.g_coupling = None
        self.permutation = None

        # Build CSCSF
        self.csf_instance = CSCSF(self.mol, self.stot, len(self.core), len(self.act))
