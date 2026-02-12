from torch import Tensor

from core.SpectralDomain import SpectralDomain
from core.HermiteBasis import hermiteBasis
import torch
from typing import Callable, List

class GHGSFMultiLobeBasis:

    def __init__(
        self,
        domain: SpectralDomain,
        centers: List[float],
        sigma: float,
        order: int
    ):
        self.m_gramInv = None
        self.m_gram = None
        self.m_basisRaw = None
        self.m_domain  = domain
        self.m_sigma   = sigma
        self.m_order   = order

        self.m_centers = torch.tensor(
            centers,
            device=domain.m_device,
            dtype=domain.m_dtype
        )

        self.m_K = len(centers)
        self.m_N = order
        self.m_M = self.m_K * self.m_N

        self.internalBuildBasis()
        self.internalComputeGram()

    def internalBuildBasis(self):
        lbda = self.m_domain.m_lambda  # [L]
        sigma = self.m_sigma
        nu = self.m_centers  # [K]

        # Expand for broadcasting
        lambda_Exp = lbda.unsqueeze(0)  # [1, L]
        nu_Exp = nu.unsqueeze(1)  # [K, 1]

        x = (lambda_Exp - nu_Exp) / sigma  # [K, L]

        # Hermite basis (batched version)
        H = hermiteBasis(self.m_N, x)  # [K, N, L]

        # --- Normalization ---
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

    def internalComputeGram(self):

        # Raw sampled basis [M, L]
        B = self.m_basisRaw

        # Match NumPy reference exactly:
        # dx = dl / sigma
        dl = self.m_domain.m_delta
        dx = dl / self.m_sigma

        # Uniform integration metric
        self.m_gram = (B @ B.T) * dx

        # Exact inverse (as in NumPy version)
        self.m_gramInv = torch.linalg.inv(self.m_gram)

    def project(self, spectrum: Tensor) -> Tensor:

        is_complex = torch.is_complex(spectrum)

        B = self.m_basisRaw
        G_inv = self.m_gramInv

        dl = self.m_domain.m_delta
        dx = dl / self.m_sigma

        if is_complex:
            B = B.to(torch.complex128)
            G_inv = G_inv.to(torch.complex128)

        # Match NumPy:
        # b = basis @ spectrum * dx
        b = (B @ spectrum) * dx

        return G_inv @ b

    def reconstruct(self, coeffs: Tensor) -> Tensor:

        if torch.is_complex(coeffs):
            B = self.m_basisRaw.to(torch.complex128)
        else:
            B = self.m_basisRaw

        return coeffs @ B



