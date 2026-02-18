import torch
from torch import Tensor

from engine.spectraloperator import SpectralOperator
from engine.ghgsfbasis import GHGSFMultiLobeBasis


class WhitenOperator:

    @staticmethod
    def create(basis: GHGSFMultiLobeBasis) -> SpectralOperator:
        """
        Whitening operator:

            α̃ = L^T α
        """

        L = basis.m_chol
        A = L.T
        b = torch.zeros(
            basis.m_M,
            device=L.device,
            dtype=L.dtype
        )

        return SpectralOperator(basis, A, b)


class UnwhitenOperator:

    @staticmethod
    def create(basis: GHGSFMultiLobeBasis) -> SpectralOperator:
        """
        Unwhitening operator:

            α = L^{-T} α̃
        """

        L = basis.m_chol
        M = basis.m_M
        device = L.device
        dtype = L.dtype

        I = torch.eye(M, device=device, dtype=dtype)

        # Solve L^T X = I  →  X = L^{-T}
        A = torch.linalg.solve_triangular(
            L.T,
            I,
            upper=True
        )

        b = torch.zeros(M, device=device, dtype=dtype)

        return SpectralOperator(basis, A, b)
