from typing import Callable

import torch
from torch import Tensor

from core.GhgsfMultiLobeBasis import GHGSFMultiLobeBasis
from core.SpectralOperator import SpectralOperator


class EmissionOperator(SpectralOperator):

    def __init__(
        self,
        basis: GHGSFMultiLobeBasis,
        emissionFn: Callable[[Tensor], Tensor]
    ):
        super().__init__(basis)
        self.m_emissionFn = emissionFn
        self.m_vector = None

    def buildVector(self):

        lbd = self.m_basis.m_domain.m_lambda
        w = self.m_basis.m_domain.m_weights
        B = self.m_basis.m_basisRaw
        G_inv = self.m_basis.m_gramInv

        spectrum = self.m_emissionFn(lbd)

        if torch.is_complex(spectrum):
            B = B.to(torch.complex128)
            w = w.to(torch.complex128)
            G_inv = G_inv.to(torch.complex128)

        b = (B * w) @ spectrum
        self.m_vector = G_inv @ b

    def emit(self) -> Tensor:

        if self.m_vector is None:
            self.buildVector()

        return self.m_vector.clone()
