import torch
from torch import Tensor
from typing import List, Literal, Optional

from engine.spectraldomain import SpectralDomain
from engine.hermitebasis import hermiteBasis


ScaleType = Literal["constant", "linear", "sqrt", "power"]


class GHGSFMultiLobeBasisFlexible:
    """
    Gaussian-Hermite Multi-Lobe Basis
    with configurable sigma growth per Hermite order.

    Scaling modes:
        - constant
        - linear
        - sqrt
        - power (requires gamma)

    Designed for large grid sweeps and geometry experiments.
    """

    def __init__(
        self,
        domain: SpectralDomain,
        centers: List[float],
        sigma_min: float,
        sigma_max: Optional[float],
        order: int,
        scale_type: ScaleType = "sqrt",
        gamma: float = 0.5
    ):
        self.m_domain = domain
        self.m_centers = torch.tensor(
            centers,
            device=domain.m_device,
            dtype=domain.m_dtype
        )
        self.m_sigma_schedule = None

        self.m_sigma_min = sigma_min
        self.m_sigma_max = sigma_max if sigma_max is not None else sigma_min
        self.m_order = order
        self.m_scale_type = scale_type
        self.m_gamma = gamma

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
    # Sigma Growth Law
    # ---------------------------------------------------------

    def _sigma_k(self, k: int, device, dtype):

        if self.m_N <= 1:
            return torch.tensor(self.m_sigma_min, device=device, dtype=dtype)

        if self.m_scale_type == "constant":
            return torch.tensor(self.m_sigma_min, device=device, dtype=dtype)

        t = torch.tensor(
            float(k) / float(self.m_N - 1),
            device=device,
            dtype=dtype
        )

        delta = self.m_sigma_max - self.m_sigma_min

        if self.m_scale_type == "linear":
            return self.m_sigma_min + delta * t

        elif self.m_scale_type == "sqrt":
            return self.m_sigma_min + delta * torch.sqrt(t)

        elif self.m_scale_type == "power":
            return self.m_sigma_min + delta * torch.pow(t, self.m_gamma)

        else:
            raise ValueError(f"Unknown scale_type: {self.m_scale_type}")

    # ---------------------------------------------------------
    # Basis Construction
    # ---------------------------------------------------------

    def _buildBasis(self):

        lbda = self.m_domain.m_lambda
        centers = self.m_centers

        device = lbda.device
        dtype = lbda.dtype

        basis_functions = []

        # -------------------------------------------------
        # Precompute sigma schedule
        # -------------------------------------------------

        sigma_list = []

        for k in range(self.m_N):
            sigma_k = self._sigma_k(k, device, dtype)
            sigma_list.append(sigma_k)

        self.m_sigma_schedule = torch.stack(sigma_list)  # [N]

        # -------------------------------------------------
        # Build basis
        # -------------------------------------------------

        for c in centers:
            for k in range(self.m_N):
                sigma_k = self.m_sigma_schedule[k]

                x = (lbda - c) / sigma_k
                x = x.unsqueeze(0)

                H = hermiteBasis(k + 1, x)

                n = torch.tensor(k, device=device, dtype=dtype)

                factorial = torch.exp(torch.lgamma(n + 1))
                sqrt_pi = torch.sqrt(torch.tensor(torch.pi, device=device, dtype=dtype))
                norm = torch.sqrt((2.0 ** n) * factorial * sqrt_pi)

                gaussian = torch.exp(-0.5 * x ** 2)

                phi_k = (H[0, k] * gaussian[0]) / norm

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

    def get_sigma_schedule(self) -> Tensor:
        return self.m_sigma_schedule.clone()

    def get_sigma_stats(self):
        sigma = self.m_sigma_schedule
        return {
            "sigma_min_actual": sigma.min().item(),
            "sigma_max_actual": sigma.max().item(),
            "sigma_mean": sigma.mean().item(),
            "sigma_std": sigma.std().item(),
            "sigma_ratio": (sigma.max() / sigma.min()).item()
        }
