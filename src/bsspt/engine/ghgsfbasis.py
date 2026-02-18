import torch
from torch import Tensor
from typing import List

from engine.spectraldomain import SpectralDomain
from engine.hermitebasis import hermiteBasis


class GHGSFMultiLobeBasis:
    """
    Gaussian-Hermite Global Spectral Function (Multi-Lobe)

    Discretized basis over spectral domain.

    Owns:
        - Raw basis matrix B      [M, L]
        - Gram matrix G           [M, M]
        - Cholesky factor L       [M, M]  (G = L L^T)

    Does NOT:
        - Store inverse
        - Perform whitening
        - Perform operator logic
    """

    def __init__(
        self,
        domain: SpectralDomain,
        centers: List[float],
        sigma: float,
        order: int
    ):
        self.m_domain = domain
        self.m_centers = torch.tensor(
            centers,
            device=domain.m_device,
            dtype=domain.m_dtype
        )

        self.m_sigma = sigma
        self.m_order = order

        self.m_K = len(centers)        # number of lobes
        self.m_N = order               # Hermite order
        self.m_M = self.m_K * self.m_N # total basis functions

        self.m_basisRaw = None  # [M, L]
        self.m_gram = None      # [M, M]
        self.m_chol = None      # [M, M]

        self._buildBasis()
        self._buildGram()
        self._buildCholesky()

    # ---------------------------------------------------------
    # Basis Construction
    # ---------------------------------------------------------

    def _buildBasis(self):

        lbda = self.m_domain.m_lambda
        sigma = self.m_sigma
        centers = self.m_centers

        lambda_exp = lbda.unsqueeze(0)       # [1, L]
        center_exp = centers.unsqueeze(1)    # [K, 1]

        x = (lambda_exp - center_exp) / sigma  # [K, L]

        H = hermiteBasis(self.m_N, x)          # [K, N, L]

        # Normalization factors
        n = torch.arange(
            self.m_N,
            device=lbda.device,
            dtype=lbda.dtype
        )

        factorial = torch.exp(torch.lgamma(n + 1))
        sqrt_pi = torch.sqrt(torch.tensor(torch.pi, device=lbda.device, dtype=lbda.dtype))

        norm = torch.sqrt((2.0 ** n) * factorial * sqrt_pi)
        norm = norm.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]

        gaussian = torch.exp(-0.5 * x ** 2).unsqueeze(1)  # [K, 1, L]

        phi = (H * gaussian) / norm  # [K, N, L]

        self.m_basisRaw = phi.reshape(self.m_M, -1)  # [M, L]

    # ---------------------------------------------------------
    # Gram Matrix
    # ---------------------------------------------------------

    def _buildGram(self):

        B = self.m_basisRaw
        w = self.m_domain.m_weights

        weighted = B * w   # broadcast over λ
        self.m_gram = weighted @ B.T  # [M, M]

    # ---------------------------------------------------------
    # Cholesky (SPD metric)
    # ---------------------------------------------------------

    def _buildCholesky(self):

        # G must be symmetric positive definite
        self.m_chol = torch.linalg.cholesky(self.m_gram)

    # ---------------------------------------------------------
    # Projection
    # ---------------------------------------------------------

    def project(self, spectrum: Tensor) -> Tensor:
        """
        Project spectrum f(λ) onto basis.

        Returns coefficients α solving:

            G α = b
        """

        B = self.m_basisRaw
        w = self.m_domain.m_weights

        if spectrum.device != B.device:
            spectrum = spectrum.to(B.device)

        if spectrum.dtype != B.dtype:
            spectrum = spectrum.to(B.dtype)

        b = (B * w) @ spectrum  # [M]

        b = b.unsqueeze(1)  # [M, 1]

        y = torch.linalg.solve_triangular(
            self.m_chol,
            b,
            upper=False
        )

        alpha = torch.linalg.solve_triangular(
            self.m_chol.T,
            y,
            upper=True
        )

        return alpha.squeeze(1)

    # ---------------------------------------------------------
    # Reconstruction
    # ---------------------------------------------------------

    def reconstruct(self, coeffs: Tensor) -> Tensor:
        """
        Reconstruct spectrum from coefficients:

            f(λ) = α^T B
        """

        B = self.m_basisRaw

        if coeffs.device != B.device:
            coeffs = coeffs.to(B.device)

        if coeffs.dtype != B.dtype:
            coeffs = coeffs.to(B.dtype)

        return coeffs @ B
