import torch
from torch import Tensor

from engine.ghgsfbasis import GHGSFMultiLobeBasis
from engine.spectralstate import SpectralState


class SpectralOperator:
    """
    Affine operator in coefficient space:

        O(α) = A α + b

    Stored as:
        m_A : [M, M]
        m_b : [M]

    Composition follows mathematical notation:

        self.compose(other) = self ∘ other
        (apply other first, then self)
    """

    def __init__(
        self,
        basis: GHGSFMultiLobeBasis,
        A: Tensor,
        b: Tensor
    ):
        self.m_basis = basis

        M = basis.m_M
        device = basis.m_basisRaw.device
        dtype = basis.m_basisRaw.dtype

        if A.shape != (M, M):
            raise ValueError("Matrix A must be shape [M, M].")

        if b.shape != (M,):
            raise ValueError("Vector b must be shape [M].")

        self.m_A = A.to(device=device, dtype=dtype)
        self.m_b = b.to(device=device, dtype=dtype)

    # ---------------------------------------------------------
    # Apply (in-place)
    # ---------------------------------------------------------

    def apply(self, state: SpectralState):
        """
        α ← A α + b
        """

        state.m_coeffs = torch.addmv(
            self.m_b,
            self.m_A,
            state.m_coeffs
        )

    # ---------------------------------------------------------
    # Composition (immediate)
    # ---------------------------------------------------------

    def compose(self, other: "SpectralOperator") -> "SpectralOperator":
        """
        Returns self ∘ other

        Apply 'other' first, then 'self'.
        """

        if self.m_basis != other.m_basis:
            raise ValueError("Basis mismatch in operator composition.")

        A_new = self.m_A @ other.m_A
        b_new = self.m_A @ other.m_b + self.m_b

        return SpectralOperator(self.m_basis, A_new, b_new)

    # ---------------------------------------------------------
    # Identity
    # ---------------------------------------------------------

    @staticmethod
    def identity(basis: GHGSFMultiLobeBasis) -> "SpectralOperator":

        M = basis.m_M
        device = basis.m_basisRaw.device
        dtype = basis.m_basisRaw.dtype

        A = torch.eye(M, device=device, dtype=dtype)
        b = torch.zeros(M, device=device, dtype=dtype)

        return SpectralOperator(basis, A, b)

    # ---------------------------------------------------------
    # Zero Operator
    # ---------------------------------------------------------

    @staticmethod
    def zero(basis: GHGSFMultiLobeBasis) -> "SpectralOperator":

        M = basis.m_M
        device = basis.m_basisRaw.device
        dtype = basis.m_basisRaw.dtype

        A = torch.zeros((M, M), device=device, dtype=dtype)
        b = torch.zeros(M, device=device, dtype=dtype)

        return SpectralOperator(basis, A, b)
