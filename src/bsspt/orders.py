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

sigma = 10.0
center = 550.0
max_order = 8

# How wide to visualize around center
view_width = 6 * sigma   # show ±6σ


# ============================================================
# Domain
# ============================================================

domain = SpectralDomain(
    lambdaMin=400.0,
    lambdaMax=700.0,
    numSamples=2048,  # higher resolution for oscillations
    device=device,
    dtype=dtype
)

lbd = domain.m_lambda


# ============================================================
# Build Basis Around Single Center
# ============================================================

x = (lbd - center) / sigma
x = x.unsqueeze(0)

H = hermiteBasis(max_order + 1, x)

n = torch.arange(max_order + 1, device=device, dtype=dtype)

factorial = torch.exp(torch.lgamma(n + 1))
sqrt_pi = torch.sqrt(torch.tensor(torch.pi, device=device, dtype=dtype))

norm = torch.sqrt((2.0 ** n) * factorial * sqrt_pi)

gaussian = torch.exp(-0.5 * x ** 2)

phi = (H[0] * gaussian) / norm.unsqueeze(-1)


# ============================================================
# Plot
# ============================================================

engine = PlotEngine(figsize=(20, 6))  # very wide

lam_cpu = lbd.detach().cpu().numpy()

for k in range(max_order + 1):
    phi_cpu = phi[k].detach().cpu().numpy()

    engine.addLine(
        lam_cpu,
        phi_cpu,
        label=f"k = {k}",
        linewidth=2.0
    )

# Zoom to region of interest
engine.setLimits(
    xlim=(center - view_width, center + view_width)
)

engine.setTitle(
    f"Gaussian–Hermite Modes (Orders 0–{max_order})\n"
    f"Center = {center} nm | σ = {sigma} nm"
)

engine.setLabels("Wavelength (nm)", "Basis Value")
engine.addLegend(location="upper right")

engine.show()
