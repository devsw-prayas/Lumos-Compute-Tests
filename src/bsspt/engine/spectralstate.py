import torch
from torch import Tensor

from engine.ghgsfbasis import GHGSFMultiLobeBasis


class SpectralState:
    """
    Coefficient-space state:

        α ∈ R^M

    Mutable.
    No algebra.
    No operator logic.
    """

    def __init__(self, basis: GHGSFMultiLobeBasis, coeffs: Tensor):
        self.m_basis = basis

        if coeffs.device != basis.m_basisRaw.device:
            coeffs = coeffs.to(basis.m_basisRaw.device)

        if coeffs.dtype != basis.m_basisRaw.dtype:
            coeffs = coeffs.to(basis.m_basisRaw.dtype)

        if coeffs.shape[0] != basis.m_M:
            raise ValueError("Coefficient dimension mismatch with basis.")

        self.m_coeffs = coeffs.clone()

    # ---------------------------------------------------------
    # Norm (Euclidean in coefficient space)
    # ---------------------------------------------------------

    def norm(self) -> Tensor:
        return torch.linalg.norm(self.m_coeffs)

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------

    def zero_(self):
        self.m_coeffs.zero_()

    # ---------------------------------------------------------
    # Clone (explicit)
    # ---------------------------------------------------------

    def clone(self):
        return SpectralState(self.m_basis, self.m_coeffs.clone())
