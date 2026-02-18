import torch
from torch import Tensor

from spectraloperator import SpectralOperator
from ghgsfbasis import GHGSFMultiLobeBasis


class DispersionOperator:

    @staticmethod
    def create(
        basis: GHGSFMultiLobeBasis,
        transferFunction: Tensor
    ) -> SpectralOperator:

        B = basis.m_basisRaw
        w = basis.m_domain.m_weights
        L = basis.m_chol

        T = transferFunction

        weighted = B * (w * T)
        M_raw = weighted @ B.T

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
