import torch
import numpy as np

from engine.spectraldomain import SpectralDomain
from engine.hermitebasis import hermiteBasis
from plotting.Plot import PlotEngine


# ============================================================
# Configuration
# ============================================================

device = torch.device("cuda")
dtype = torch.float64

center = 550.0
max_order = 8

sigma0 = 3.0        # narrowest (order 0)
alpha = 2.0         # growth per order

# how wide to view
view_width = 60.0


# ============================================================
# Domain
# ============================================================

domain = SpectralDomain(
    lambdaMin=400.0,
    lambdaMax=700.0,
    numSamples=2048,
    device=device,
    dtype=dtype
)

lbd = domain.m_lambda
lam_cpu = lbd.detach().cpu().numpy()


# ============================================================
# Plot
# ============================================================

engine = PlotEngine(figsize=(20, 6))

for k in range(max_order + 1):

    sigma_k = sigma0 + alpha * k

    x = (lbd - center) / sigma_k
    x = x.unsqueeze(0)

    # Build only up to k, then take kth slice
    H = hermiteBasis(k + 1, x)

    n = torch.tensor(k, device=device, dtype=dtype)

    factorial = torch.exp(torch.lgamma(n + 1))
    sqrt_pi = torch.sqrt(torch.tensor(torch.pi, device=device, dtype=dtype))

    norm = torch.sqrt((2.0 ** n) * factorial * sqrt_pi)

    gaussian = torch.exp(-0.5 * x ** 2)

    phi_k = (H[0, k] * gaussian[0]) / norm

    phi_cpu = phi_k.detach().cpu().numpy()

    engine.addLine(
        lam_cpu,
        phi_cpu,
        label=f"k = {k}, σ = {sigma_k:.1f}",
        linewidth=2.0
    )

engine.setLimits(
    xlim=(center - view_width, center + view_width)
)

engine.setTitle(
    f"Gaussian–Hermite Modes (Variable σ per Order)\n"
    f"σ₀ = {sigma0}, growth = {alpha} per order"
)

engine.setLabels("Wavelength (nm)", "Basis Value")
engine.addLegend(location="upper right")

engine.show()
