import torch
from torch import Tensor
from typing import Callable

from spectraloperator import SpectralOperator
from ghgsfbasis import GHGSFMultiLobeBasis


class EmissionOperator:

    @staticmethod
    def create(
        basis: GHGSFMultiLobeBasis,
        emissionFn: Callable[[Tensor], Tensor]
    ) -> SpectralOperator:

        B = basis.m_basisRaw
        w = basis.m_domain.m_weights
        L = basis.m_chol

        lbda = basis.m_domain.m_lambda
        spectrum = emissionFn(lbda)

        raw = (B * w) @ spectrum

        # Solve G b = raw
        y = torch.linalg.solve_triangular(
            L,
            raw,
            upper=False
        )

        b = torch.linalg.solve_triangular(
            L.T,
            y,
            upper=True
        )

        A = torch.zeros(
            (basis.m_M, basis.m_M),
            device=b.device,
            dtype=b.dtype
        )

        return SpectralOperator(basis, A, b)
