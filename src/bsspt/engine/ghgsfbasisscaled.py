import torch
from torch import Tensor
from typing import List

from engine.spectraldomain import SpectralDomain
from engine.hermitebasis import hermiteBasis


class GHGSFMultiLobeBasisScaled:
    """
    Gaussian-Hermite Multi-Lobe Basis
    with sqrt growth of sigma per Hermite order.

        sigma_k = sigma_min + beta * sqrt(k)

    Fully compatible with original projection + reconstruction.
    """

    def __init__(
        self,
        domain: SpectralDomain,
        centers: List[float],
        sigma_min: float,
        sigma_max: float,
        order: int
    ):
        self.m_domain = domain
        self.m_centers = torch.tensor(
            centers,
            device=domain.m_device,
            dtype=domain.m_dtype
        )

        self.m_sigma_min = sigma_min
        self.m_sigma_max = sigma_max
        self.m_order = order

        self.m_K = len(centers)
        self.m_N = order
        self.m_M = self.m_K * self.m_N

        self.m_basisRaw = None
        self.m_gram = None
        self.m_chol = None

        self._buildBasis()
        self._buildGram()
        self._buildCholesky()

    # ---------------------------------------------------------
    # Basis Construction
    # ---------------------------------------------------------

    def _buildBasis(self):

        lbda = self.m_domain.m_lambda
        centers = self.m_centers

        device = lbda.device
        dtype = lbda.dtype

        basis_functions = []

        # Precompute sigma scaling
        if self.m_N > 1:
            beta = (self.m_sigma_max - self.m_sigma_min) / torch.sqrt(
                torch.tensor(float(self.m_N - 1), device=device, dtype=dtype)
            )
        else:
            beta = torch.tensor(0.0, device=device, dtype=dtype)

        for c in centers:

            for k in range(self.m_N):
                # ---- sqrt growth sigma ----
                sigma_k = self.m_sigma_min + beta * torch.sqrt(
                    torch.tensor(float(k), device=device, dtype=dtype)
                )

                # ---- build x for this (c, k) ----
                x = (lbda - c) / sigma_k  # [L]
                x = x.unsqueeze(0)  # [1, L]

                # Hermite up to k
                H = hermiteBasis(k + 1, x)  # [1, k+1, L]

                n = torch.tensor(k, device=device, dtype=dtype)

                factorial = torch.exp(torch.lgamma(n + 1))
                sqrt_pi = torch.sqrt(torch.tensor(torch.pi, device=device, dtype=dtype))

                norm = torch.sqrt((2.0 ** n) * factorial * sqrt_pi)

                gaussian = torch.exp(-0.5 * x ** 2)

                phi_k = (H[0, k] * gaussian[0]) / norm  # [L]

                basis_functions.append(phi_k)

        self.m_basisRaw = torch.stack(basis_functions, dim=0)

    # ---------------------------------------------------------
    # Gram Matrix
    # ---------------------------------------------------------

    def _buildGram(self):

        B = self.m_basisRaw
        w = self.m_domain.m_weights

        weighted = B * w
        self.m_gram = weighted @ B.T

    # ---------------------------------------------------------
    # Cholesky
    # ---------------------------------------------------------

    def _buildCholesky(self):

        self.m_chol = torch.linalg.cholesky(self.m_gram)

    # ---------------------------------------------------------
    # Projection
    # ---------------------------------------------------------

    def project(self, spectrum: Tensor) -> Tensor:

        B = self.m_basisRaw
        w = self.m_domain.m_weights

        if spectrum.device != B.device:
            spectrum = spectrum.to(B.device)

        if spectrum.dtype != B.dtype:
            spectrum = spectrum.to(B.dtype)

        b = (B * w) @ spectrum
        b = b.unsqueeze(1)

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

        B = self.m_basisRaw

        if coeffs.device != B.device:
            coeffs = coeffs.to(B.device)

        if coeffs.dtype != B.dtype:
            coeffs = coeffs.to(B.dtype)

        return coeffs @ B
