import torch
from torch import Tensor
from typing import Callable

from spectraloperator import SpectralOperator
from ghgsfbasis import GHGSFMultiLobeBasis


class AbsorptionOperator:

    @staticmethod
    def create(
        basis: GHGSFMultiLobeBasis,
        sigmaA: Callable[[Tensor], Tensor],
        distance: float
    ) -> SpectralOperator:

        B = basis.m_basisRaw
        w = basis.m_domain.m_weights
        L = basis.m_chol

        lbda = basis.m_domain.m_lambda

        T = torch.exp(-sigmaA(lbda) * distance)

        weighted = B * (w * T)
        M_raw = weighted @ B.T   # [M, M]

        # Solve G A = M_raw using Cholesky
        Y = torch.linalg.solve_triangular(
            L,
            M_raw,
            upper=False
        )

        A = torch.linalg.solve_triangular(
            L.T,
            Y,
            upper=True
        )

        b = torch.zeros(
            basis.m_M,
            device=A.device,
            dtype=A.dtype
        )

        return SpectralOperator(basis, A, b)
