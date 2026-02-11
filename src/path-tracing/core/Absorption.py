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
        B = self.m_basis.m_basisTight

        T = torch.exp(-self.m_sigmaA(lbd) * self.m_distance)

        weighted = B * (w * T)

        self.m_matrix = weighted @ B.T
