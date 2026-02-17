import torch
from torch import Tensor

from core.SpectralOperator import SpectralOperator
from core.GhgsfMultiLobeBasis import GHGSFMultiLobeBasis


class DispersionOperator(SpectralOperator):

    def __init__(
        self,
        basis: GHGSFMultiLobeBasis,
        transferFunction: Tensor  # T(λ), complex allowed
    ):
        super().__init__(basis)

        self.m_transfer = transferFunction  # shape [L]

    def buildMatrix(self):

        B = self.m_basis.m_basisRaw          # [M, L]
        G = self.m_basis.m_gram              # [M, M]
        w = self.m_basis.m_domain.m_weights  # [L]

        T = self.m_transfer                  # [L]

        # --- Type alignment ---
        if torch.is_complex(T):
            B = B.to(T.dtype)
            G = G.to(T.dtype)

        # --- Weighted multiplication ---
        # Implements:
        #   M_raw[i,j] = ∫ φ_i(λ) T(λ) φ_j(λ) dλ
        #
        # Discrete:
        #   sum_k φ_i(λ_k) T(λ_k) φ_j(λ_k) w_k

        weighted = B * (w * T)               # broadcast over λ
        M_raw = weighted @ B.T               # [M, M]

        # --- Galerkin projection ---
        # Solve G X = M_raw  →  X = G^{-1} M_raw
        #
        # Do NOT use inverse explicitly

        if torch.is_complex(M_raw):
            G = G.to(M_raw.dtype)

        self.m_matrix = torch.linalg.solve(G, M_raw)
