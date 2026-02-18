import torch
import numpy as np

from engine.ghgsfbasisscaled import GHGSFMultiLobeBasisScaled
from engine.spectraldomain import SpectralDomain
from engine.ghgsfbasis import GHGSFMultiLobeBasis
from engine.spectralstate import SpectralState
from engine.whitening import WhitenOperator

from plotting.Plot import MultiPanelEngine


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


# ============================================================
# Utility
# ============================================================

def normalize(S):
    return S / domain.integrate(torch.abs(S))


# ============================================================
# Real Spectral Cases
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


cases = {
    "D65": normalize(d65(lbd)),
    "LED": normalize(leds(lbd)),
    "Blackbody": normalize(blackbody(lbd)),
    "Step": normalize(step_spectrum(lbd)),
    "Laser": normalize(laser_spike(lbd)),
    "Comb": normalize(comb_spectrum(lbd)),
}


# ============================================================
# Stress Sweep
# ============================================================

sigma_min = 5
sigma_max = 9
orders = range(6, 9)
lobe_counts = range(6, 9)

for order in orders:
    for n_lobes in lobe_counts:

        centers = torch.linspace(420.0, 680.0, n_lobes).tolist()

        basis = GHGSFMultiLobeBasisScaled(
            domain=domain,
            centers=centers,
            sigma_min=sigma_min,  # narrow capture
            sigma_max=sigma_max,  # smooth correction
            order=order
        )

        cond = torch.linalg.cond(basis.m_gram).item()

        print(f"Running {n_lobes} lobes × order {order} | cond={cond:.2e}")

        engine = MultiPanelEngine(
            nrows=3,
            ncols=2,
            figsize=(14, 12),
            compact=True
        )

        # Build whitening operator once per basis
        W = WhitenOperator.create(basis)

        for i, (name, S) in enumerate(cases.items()):

            # ------------------------------------------------
            # Projection & Reconstruction
            # ------------------------------------------------

            coeffs = basis.project(S)
            R = basis.reconstruct(coeffs)

            # ------------------------------------------------
            # Raw State
            # ------------------------------------------------

            state_raw = SpectralState(basis, coeffs)

            # ------------------------------------------------
            # Whitened State
            # ------------------------------------------------

            state_white = state_raw.clone()
            W.apply(state_white)

            alpha_raw = state_raw.m_coeffs
            alpha_white = state_white.m_coeffs

            # ------------------------------------------------
            # Error Metrics
            # ------------------------------------------------

            L2 = torch.sqrt(
                domain.integrate((S - R)**2)
            ).item()

            Linf = torch.max(torch.abs(S - R)).item()

            energy_err = torch.abs(
                domain.integrate(S) -
                domain.integrate(R)
            ).item()

            max_alpha_raw = torch.max(torch.abs(alpha_raw)).item()
            max_alpha_white = torch.max(torch.abs(alpha_white)).item()

            norm_raw = torch.linalg.norm(alpha_raw).item()
            norm_white = torch.linalg.norm(alpha_white).item()

            # ------------------------------------------------
            # Plot
            # ------------------------------------------------

            panel = engine.getPanel(i)

            lam_cpu = lbd.detach().cpu().numpy()
            S_cpu = S.detach().cpu().numpy()
            R_cpu = R.detach().cpu().numpy()

            panel.addLine(lam_cpu, S_cpu, linewidth=2.2, label="Original")
            panel.addLine(lam_cpu, R_cpu, linestyle="--", linewidth=2.0, label="Reconstruction")

            panel.setTitle(name)
            panel.setLabels("Wavelength (nm)", "Power")

            panel.annotateMetricsBlock({
                "L2": f"{L2:.2e}",
                "L∞": f"{Linf:.2e}",
                "ΔE": f"{energy_err:.2e}",
                "max|α|": f"{max_alpha_raw:.2e}",
                "max|α̃|": f"{max_alpha_white:.2e}",
                "||α||": f"{norm_raw:.2e}",
                "||α̃||": f"{norm_white:.2e}"
            }, position='upper left')

        engine.addLegendOnlyFirst()
        engine.applyDenseLayout()
        engine.applyPublicationPreset()

        engine.setMainTitle(
            f"GHGSF Stress Test — {n_lobes} lobes × order {order}\n"
            f"σ = {sigma_min} nm  to {sigma_max} nm| Gram cond = {cond:.2e}"
        )

        engine.show()
