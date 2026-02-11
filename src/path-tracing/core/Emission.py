from typing import Callable

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
        B = self.m_basis.m_basisTight

        spectrum = self.m_emissionFn(lbd)

        self.m_vector = (B * w) @ spectrum

    def emit(self) -> Tensor:

        if self.m_vector is None:
            self.buildVector()

        return self.m_vector.clone()
