import torch
import numpy as np

import prrequisite
from core.SpectralDomain import SpectralDomain
from core.GhgsfMultiLobeBasis import GHGSFMultiLobeBasis
from plotting.Plot import MultiPanelEngine

prrequisite.SandboxRuntime.bootstrap()


# ============================================================
# Domain
# ============================================================

domain = SpectralDomain(
    lambdaMin=400.0,
    lambdaMax=700.0,
    numSamples=256,
    device=torch.device("cuda"),
    dtype=torch.float64
)

lbd = domain.m_lambda

torch.set_default_dtype(torch.float64)

# ============================================================
# Utility
# ============================================================

def normalize(S):
    return S / domain.integrate(torch.abs(S))

# ============================================================
# Spectral Cases (Torch)
# ============================================================

def d65(l):
    return (
        torch.exp(-0.5 * ((l - 445.0) / 40.0)**2) +
        0.9 * torch.exp(-0.5 * ((l - 540.0) / 50.0)**2) +
        0.6 * torch.exp(-0.5 * ((l - 610.0) / 60.0)**2)
    )

def leds(l):
    return (
        1.2 * torch.exp(-0.5 * ((l - 450.0) / 6.0)**2) +
        1.0 * torch.exp(-0.5 * ((l - 530.0) / 5.0)**2) +
        0.9 * torch.exp(-0.5 * ((l - 625.0) / 7.0)**2)
    )

def blackbody(l, T=6500.0):
    c2 = 1.4388e7
    return (l**-5) / (torch.exp(c2 / (l * T)) - 1.0)

def beer_lambert(l):
    return d65(l) * torch.exp(-0.015 * (l - 500.0))

def reflectance(l):
    R = (
        0.7
        - 0.4 * torch.exp(-0.5 * ((l - 500.0) / 30.0)**2)
        - 0.3 * torch.exp(-0.5 * ((l - 650.0) / 20.0)**2)
    )
    return torch.clamp(R, 0.0, 1.0)

def iridescence(l):
    env = torch.exp(-0.5 * ((l - 550.0) / 60.0)**2)
    osc = 0.5 * (1.0 + torch.cos(0.25 * l))
    return env * osc

def step_spectrum(l):
    return torch.where(l < 550.0, 1.0, 0.25)

def laser_spike(l):
    return torch.exp(-0.5 * ((l - 532.0) / 1.0)**2)

def comb_spectrum(l):
    centers = torch.linspace(420.0, 680.0, 12, device=l.device, dtype=l.dtype)
    S = torch.zeros_like(l)
    for c in centers:
        S += torch.exp(-0.5 * ((l - c) / 1.5)**2)
    return S

def dispersion(l):
    base = leds(l)
    phase = 0.002 * (l - 555.0)**2
    return base * torch.exp(1j * phase)

cases = {
    "D65": normalize(d65(lbd)),
    "LED": normalize(leds(lbd)),
    "Blackbody": normalize(blackbody(lbd)),
    "Beer–Lambert": normalize(beer_lambert(lbd)),
    "Reflectance": normalize(reflectance(lbd)),
    "Iridescence": normalize(iridescence(lbd)),
    "Step": normalize(step_spectrum(lbd)),
    "Laser": normalize(laser_spike(lbd)),
    "Comb": normalize(comb_spectrum(lbd)),
    "Dispersion": normalize(dispersion(lbd)),
}

# ============================================================
# Stress Sweep
# ============================================================

sigma = 8.0
orders = range(6, 9)
lobe_counts = range(6, 9)

for order in orders:
    for n_lobes in lobe_counts:

        centers = torch.linspace(420.0, 680.0, n_lobes).tolist()

        basis = GHGSFMultiLobeBasis(
            domain=domain,
            centers=centers,
            sigma=sigma,
            order=order
        )

        cond = torch.linalg.cond(basis.m_gram).item()

        print(f"Running {n_lobes} lobes × order {order} | cond={cond:.2e}")

        engine = MultiPanelEngine(
            nrows=5,
            ncols=2,
            figsize=(14, 18),
            compact=True
        )

        for i, (name, S) in enumerate(cases.items()):

            coeffs = basis.project(S)
            R = basis.reconstruct(coeffs)

            L2 = torch.sqrt(
                domain.integrate(torch.abs(S - R)**2)
            ).item()

            Linf = torch.max(torch.abs(S - R)).item()

            energy_err = torch.abs(
                domain.integrate(torch.abs(S)) -
                domain.integrate(torch.abs(R))
            ).item()

            max_alpha = torch.max(torch.abs(coeffs)).item()

            panel = engine.getPanel(i)

            lam_cpu = lbd.detach().cpu().numpy()
            S_cpu = torch.abs(S).detach().cpu().numpy()
            R_cpu = torch.abs(R).detach().cpu().numpy()

            panel.addLine(lam_cpu, S_cpu, linewidth=2.2, label="Original")
            panel.addLine(lam_cpu, R_cpu, linestyle="--", linewidth=2.0, label="Reconstruction")

            panel.setTitle(name)
            panel.setLabels("Wavelength (nm)", "Power")

            panel.annotateMetricsBlock({
                "L2": f"{L2:.2e}",
                "L∞": f"{Linf:.2e}",
                "ΔE": f"{energy_err:.2e}",
                "max|α|": f"{max_alpha:.2e}"
            }, position='upper left')

        engine.addLegendOnlyFirst()

        engine.applyDenseLayout()

        engine.applyPublicationPreset()

        engine.setMainTitle(
            f"GHGSF Stress Test — {n_lobes} lobes × order {order}\n"
            f"σ = {sigma} nm | Gram cond = {cond:.2e}"
        )

        engine.show()