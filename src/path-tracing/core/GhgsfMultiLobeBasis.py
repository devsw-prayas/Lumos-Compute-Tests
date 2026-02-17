from torch import Tensor
from core.SpectralDomain import SpectralDomain
from core.HermiteBasis import hermiteBasis
import torch
from typing import List


class GHGSFMultiLobeBasis:

    def __init__(
        self,
        domain: SpectralDomain,
        centers: List[float],
        sigma: float,
        order: int
    ):
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

        self.m_basisRaw = None
        self.m_gram = None
        self.m_gramInv = None

        self.m_L = None
        self.m_Linv = None
        self.m_LT = None
        self.m_LinvT = None

        self.internalBuildBasis()
        self.internalComputeGram()
        self.internalComputeWhitening()

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------

    def _match(self, A: Tensor, B: Tensor) -> Tensor:
        """
        Cast A to match dtype and device of B.
        """
        return A.to(dtype=B.dtype, device=B.device)

    # ---------------------------------------------------------
    # Basis Construction
    # ---------------------------------------------------------

    def internalBuildBasis(self):

        lbda = self.m_domain.m_lambda
        sigma = self.m_sigma
        nu = self.m_centers

        lambda_exp = lbda.unsqueeze(0)
        nu_exp = nu.unsqueeze(1)

        x = (lambda_exp - nu_exp) / sigma

        H = hermiteBasis(self.m_N, x)

        n = torch.arange(
            self.m_N,
            device=lbda.device,
            dtype=lbda.dtype
        )

        factorial = torch.exp(torch.lgamma(n + 1))
        sqrt_pi = torch.sqrt(torch.tensor(torch.pi, device=lbda.device, dtype=lbda.dtype))

        norm = torch.sqrt((2.0 ** n) * factorial * sqrt_pi)
        norm = norm.unsqueeze(0).unsqueeze(-1)

        gaussian = torch.exp(-0.5 * x ** 2).unsqueeze(1)

        phi = (H * gaussian) / norm

        self.m_basisRaw = phi.reshape(self.m_M, -1)

    # ---------------------------------------------------------
    # Gram
    # ---------------------------------------------------------

    def internalComputeGram(self):

        B = self.m_basisRaw

        w = self.m_domain.m_weights
        weighted = B * w
        self.m_gram = weighted @ B.T

        # Prefer solve over explicit inverse (more stable)
        I = torch.eye(
            self.m_gram.shape[0],
            device=self.m_gram.device,
            dtype=self.m_gram.dtype
        )

        self.m_gramInv = torch.linalg.solve(self.m_gram, I)

    # ---------------------------------------------------------
    # Projection
    # ---------------------------------------------------------

    def project(self, spectrum: Tensor) -> Tensor:

        if not isinstance(spectrum, torch.Tensor):
            raise TypeError("Spectrum must be a torch.Tensor")

        B = self.m_basisRaw
        G_inv = self.m_gramInv

        # Device alignment
        if spectrum.device != B.device:
            spectrum = spectrum.to(B.device)

        # Complex branch
        if torch.is_complex(spectrum):
            B = self._match(B, spectrum)
            G_inv = self._match(G_inv, spectrum)
        else:
            if spectrum.dtype != B.dtype:
                spectrum = spectrum.to(B.dtype)

        w = self.m_domain.m_weights
        b = (B * w) @ spectrum
        G = self.m_gram

        if torch.is_complex(b):
            G = G.to(b.dtype)

        return torch.linalg.solve(G, b)

    # ---------------------------------------------------------
    # Reconstruction
    # ---------------------------------------------------------

    def reconstruct(self, coeffs: Tensor) -> Tensor:

        if not isinstance(coeffs, torch.Tensor):
            raise TypeError("Coefficients must be a torch.Tensor")

        B = self.m_basisRaw

        if coeffs.device != B.device:
            coeffs = coeffs.to(B.device)

        if torch.is_complex(coeffs):
            B = self._match(B, coeffs)
        elif coeffs.dtype != B.dtype:
            coeffs = coeffs.to(B.dtype)

        return coeffs @ B

    # ---------------------------------------------------------
    # Whitening
    # ---------------------------------------------------------

    def internalComputeWhitening(self):

        # Cholesky (G = L L^T)
        self.m_L = torch.linalg.cholesky(self.m_gram)

        self.m_Linv = torch.linalg.solve(
            self.m_L,
            torch.eye(
                self.m_L.shape[0],
                device=self.m_L.device,
                dtype=self.m_L.dtype
            )
        )

        self.m_LT = self.m_L.T
        self.m_LinvT = self.m_Linv.T

    def projectWhitened(self, spectrum: Tensor) -> Tensor:

        alpha = self.project(spectrum)

        LT = self._match(self.m_LT, alpha)

        return LT @ alpha

    def reconstructWhitened(self, alpha_tilde: Tensor) -> Tensor:

        if not isinstance(alpha_tilde, torch.Tensor):
            raise TypeError("Whitened coefficients must be a torch.Tensor")

        LinvT = self._match(self.m_LinvT, alpha_tilde)

        alpha = LinvT @ alpha_tilde

        return self.reconstruct(alpha)
