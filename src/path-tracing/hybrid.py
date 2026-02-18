import torch
import math
import numpy as np
import prrequisite

from core.SpectralDomain import SpectralDomain
from core.GhgsfMultiLobeBasis import GHGSFMultiLobeBasis
from plotting.Plot import MultiPanelEngine

prrequisite.SandboxRuntime.bootstrap()

# ============================================================
# Setup
# ============================================================

torch.set_default_dtype(torch.float64)
device = torch.get_default_device()

domain = SpectralDomain(
    lambdaMin=400.0,
    lambdaMax=700.0,
    numSamples=512,
    device=device,
    dtype=torch.get_default_dtype()
)

lam = domain.m_lambda
L = lam[-1] - lam[0]

# ============================================================
# Spectral Test Cases
# ============================================================

def laser_spike(l):
    return torch.exp(-0.5 * ((l - 532.0) / 1.0)**2)

def comb_spectrum(l):
    centers = torch.linspace(420.0, 680.0, 18, device=l.device)
    S = torch.zeros_like(l)
    for c in centers:
        S += torch.exp(-0.5 * ((l - c) / 1.2)**2)
    return S

def extreme_iridescence(l):
    env = torch.exp(-0.5 * ((l - 550.0) / 80.0)**2)
    return env * (1.0 + torch.cos(0.6 * l))

def dispersion(l):
    base = laser_spike(l)
    phase = 0.003 * (l - 550.0)**2
    return base * torch.exp(1j * phase)

def normalize(S):
    return S / domain.integrate(torch.abs(S))

cases = {
    "Laser": normalize(laser_spike(lam)),
    "Comb": normalize(comb_spectrum(lam)),
    "Iridescence": normalize(extreme_iridescence(lam)),
    "Dispersion": normalize(dispersion(lam)),
}

# ============================================================
# Fourier Basis
# ============================================================

def build_fourier_basis(domain, max_mode):

    # Keep lambda REAL for domain length
    l_real = domain.m_lambda
    L = (l_real[-1] - l_real[0]).item()

    # Convert to complex only for exponentials
    l = l_real.to(torch.complex64)
    x = (l - l_real[0]) / L

    modes = torch.arange(
        -max_mode,
         max_mode + 1,
        device=l.device
    )

    basis = []
    norm = math.sqrt(L)  # real

    for k in modes:
        phi = torch.exp(2j * math.pi * k * x)
        phi = phi / norm
        basis.append(phi)

    return torch.stack(basis, dim=0)


# ============================================================
# Hybrid Builder
# ============================================================

def build_hybrid_basis(domain, ghgsf, max_mode):

    B1 = ghgsf.m_basisRaw.to(torch.complex64)
    B2 = build_fourier_basis(domain, max_mode)

    B = torch.cat([B1, B2], dim=0)

    w = domain.m_weights.to(torch.complex64)
    G = (B * w) @ torch.conj(B.T)

    # Regularize slightly for safety
    G += 1e-10 * torch.eye(G.shape[0], device=B.device)

    return B, G

# ============================================================
# GHGSF Setup
# ============================================================

sigma = 7.6
centers = torch.linspace(420.0, 680.0, 8).tolist()

ghgsf = GHGSFMultiLobeBasis(
    domain=domain,
    centers=centers,
    sigma=sigma,
    order=6
)

# ============================================================
# Build Hybrid
# ============================================================

max_fourier_mode = 20
B_hybrid, G_hybrid = build_hybrid_basis(domain, ghgsf, max_fourier_mode)

cond_hybrid = torch.linalg.cond(G_hybrid).item()
print(f"Hybrid Gram condition number: {cond_hybrid:.2e}")

# ============================================================
# Projection Utility
# ============================================================

def project(B, G, S):

    S = S.to(torch.complex64)
    w = domain.m_weights.to(torch.complex64)

    b = (B * w) @ S
    alpha = torch.linalg.solve(G, b)
    return alpha

def reconstruct(B, alpha):
    return alpha @ B

# ============================================================
# Plot Comparison
# ============================================================

engine = MultiPanelEngine(
    nrows=2,
    ncols=2,
    figsize=(12, 8),
    compact=True
)

for i, (name, S) in enumerate(cases.items()):

    # ---------------- GHGSF ----------------
    coeff_g = ghgsf.project(S)
    R_g = ghgsf.reconstruct(coeff_g)

    L2_g = torch.sqrt(
        domain.integrate(torch.abs(S - R_g)**2)
    ).item()

    # ---------------- Fourier Only ----------------
    B_fourier = build_fourier_basis(domain, max_fourier_mode)
    w = domain.m_weights.to(torch.complex64)
    G_fourier = (B_fourier * w) @ torch.conj(B_fourier.T)
    G_fourier += 1e-10 * torch.eye(G_fourier.shape[0], device=device)

    coeff_f = project(B_fourier, G_fourier, S)
    R_f = reconstruct(B_fourier, coeff_f)

    L2_f = torch.sqrt(
        domain.integrate(torch.abs(S - R_f)**2)
    ).item()

    # ---------------- Hybrid ----------------
    coeff_h = project(B_hybrid, G_hybrid, S)
    R_h = reconstruct(B_hybrid, coeff_h)

    L2_h = torch.sqrt(
        domain.integrate(torch.abs(S - R_h)**2)
    ).item()

    # ---------------- Plot ----------------
    panel = engine.getPanel(i)

    lam_cpu = lam.detach().cpu().numpy()

    panel.addLine(lam_cpu,
                  torch.abs(S).cpu().numpy(),
                  linewidth=2.2,
                  label="Original")

    panel.addLine(lam_cpu,
                  torch.abs(R_g).cpu().numpy(),
                  linestyle="--",
                  label=f"GHGSF ({L2_g:.2e})")

    panel.addLine(lam_cpu,
                  torch.abs(R_f).cpu().numpy(),
                  linestyle=":",
                  label=f"Fourier ({L2_f:.2e})")

    panel.addLine(lam_cpu,
                  torch.abs(R_h).cpu().numpy(),
                  linestyle="-.",
                  label=f"Hybrid ({L2_h:.2e})")

    panel.setTitle(name)
    panel.setLabels("Wavelength (nm)", "Amplitude")

engine.addLegendOnlyFirst()
engine.applyDenseLayout()
engine.show()
