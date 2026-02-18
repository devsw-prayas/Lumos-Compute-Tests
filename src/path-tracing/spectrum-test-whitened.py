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
    device=torch.get_default_device(),
    dtype=torch.get_default_dtype()
)

lbd = domain.m_lambda

# ============================================================
# Utility
# ============================================================

def normalize(S):
    return S / domain.integrate(torch.abs(S))

# ============================================================
# Spectral Cases
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

sigma = 7.6
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

        print(f"\nRunning {n_lobes} lobes × order {order} | cond={cond:.2e}")

        engine = MultiPanelEngine(
            nrows=5,
            ncols=2,
            figsize=(14, 18),
            compact=True
        )

        for i, (name, S) in enumerate(cases.items()):

            # ==========================
            # RAW PROJECTION
            # ==========================
            coeffs_raw = basis.project(S)
            R_raw = basis.reconstruct(coeffs_raw)

            L2_raw = torch.sqrt(
                domain.integrate(torch.abs(S - R_raw)**2)
            ).item()

            Linf_raw = torch.max(torch.abs(S - R_raw)).item()

            energy_raw = torch.abs(
                domain.integrate(torch.abs(S)) -
                domain.integrate(torch.abs(R_raw))
            ).item()

            # Hermitian coefficient energy
            G = basis.m_gram.to(coeffs_raw.dtype)

            energy_coeff_raw = torch.real(
                torch.conj(coeffs_raw) @ (G @ coeffs_raw)
            ).item()

            # ==========================
            # WHITENED PROJECTION
            # ==========================
            coeffs_white = basis.projectWhitened(S)
            R_white = basis.reconstructWhitened(coeffs_white)

            L2_white = torch.sqrt(
                domain.integrate(torch.abs(S - R_white)**2)
            ).item()

            Linf_white = torch.max(torch.abs(S - R_white)).item()

            energy_white = torch.abs(
                domain.integrate(torch.abs(S)) -
                domain.integrate(torch.abs(R_white))
            ).item()

            # Euclidean energy in whitened space
            energy_coeff_white = torch.real(
                torch.conj(coeffs_white) @ coeffs_white
            ).item()

            energy_coeff_diff = abs(
                energy_coeff_raw - energy_coeff_white
            )

            # ==========================
            # Plotting
            # ==========================
            panel = engine.getPanel(i)

            lam_cpu = lbd.detach().cpu().numpy()
            S_cpu = torch.abs(S).detach().cpu().numpy()
            R_raw_cpu = torch.abs(R_raw).detach().cpu().numpy()
            R_white_cpu = torch.abs(R_white).detach().cpu().numpy()

            panel.addLine(lam_cpu, S_cpu,
                          linewidth=2.2,
                          label="Original")

            panel.addLine(lam_cpu, R_raw_cpu,
                          linestyle="--",
                          linewidth=2.0,
                          label="Raw")

            panel.addLine(lam_cpu, R_white_cpu,
                          linestyle=":",
                          linewidth=1.8,
                          color="#FF3B3B",  # bright red
                          alpha=0.9,
                          label="Whitened")

            panel.setTitle(name)
            panel.setLabels("Wavelength (nm)", "Power")

            panel.annotateMetricsBlock({
                "L2 raw": f"{L2_raw:.2e}",
                "L2 white": f"{L2_white:.2e}",
                "ΔE raw": f"{energy_raw:.2e}",
                "ΔE white": f"{energy_white:.2e}",
                "Ec raw": f"{energy_coeff_raw:.2e}",
                "Ec white": f"{energy_coeff_white:.2e}",
                "ΔEc": f"{energy_coeff_diff:.2e}",
            }, position='upper left')

        engine.addLegendOnlyFirst()
        engine.applyDenseLayout()
        engine.applyPublicationPreset()

        engine.setMainTitle(
            f"GHGSF Raw vs Whitened Stress Test — "
            f"{n_lobes} lobes × order {order}\n"
            f"σ = {sigma} nm | Gram cond = {cond:.2e}"
        )

        engine.show()
