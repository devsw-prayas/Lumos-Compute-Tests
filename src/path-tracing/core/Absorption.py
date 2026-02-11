from typing import Callable

import torch
from torch import Tensor

from core.GhgsfMultiLobeBasis import GHGSFMultiLobeBasis
from core.SpectralOperator import SpectralOperator


class AbsorptionOperator(SpectralOperator):

    def __init__(
        self,
        basis: GHGSFMultiLobeBasis,
        sigmaA: Callable[[Tensor], Tensor],
        distance: float
    ):
        super().__init__(basis)
        self.m_sigmaA = sigmaA
        self.m_distance = distance

    def buildMatrix(self):
        lbd = self.m_basis.m_domain.m_lambda
        w = self.m_basis.m_domain.m_weights

        B = self.m_basis.m_basisRaw
        G_inv = self.m_basis.m_gramInv

        T = torch.exp(-self.m_sigmaA(lbd) * self.m_distance)

        if torch.is_complex(T):
            B = B.to(torch.complex128)
            w = w.to(torch.complex128)
            G_inv = G_inv.to(torch.complex128)

        weighted = B * (w * T)
        M_raw = weighted @ B.T

        self.m_matrix = G_inv @ M_raw