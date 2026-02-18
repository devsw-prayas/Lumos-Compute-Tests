import torch
import numpy as np

from core.SpectralDomain import SpectralDomain
from core.GhgsfMultiLobeBasis import GHGSFMultiLobeBasis
from plotting.Plot import PlotEngine

# ============================================================
# Setup
# ============================================================

torch.set_default_dtype(torch.float64)
torch.set_default_device(torch.device("cuda"))

device = torch.get_default_device()
dtype  = torch.get_default_dtype()

domain = SpectralDomain(
    lambdaMin=400.0,
    lambdaMax=700.0,
    numSamples=4096,
    device=device,
    dtype=dtype
)

lam = domain.m_lambda

# ============================================================
# Initial spectrum
# ============================================================

def initial_spectrum(l):
    return (
        0.8 * torch.exp(-0.5 * ((l - 460.0) / 60.0)**2) +
        0.9 * torch.exp(-0.5 * ((l - 560.0) / 70.0)**2)
    )

S_init = initial_spectrum(lam)
S_init /= domain.integrate(S_init)

# ============================================================
# GHGSF Basis (48D)
# ============================================================

sigma = 7.5
lobes = torch.linspace(405.0, 695.0, 8).tolist()
order = 7

basis = GHGSFMultiLobeBasis(
    domain=domain,
    centers=lobes,
    sigma=sigma,
    order=order
)

alpha = basis.projectWhitened(S_init)

# ============================================================
# Extreme Oscillatory Operator
# ============================================================

def oscillatory_operator(l):
    A = 0.9
    B = 0.7
    omega = 0.6
    C     = 8000.0

    f = (1.0 + A * torch.cos(omega * l))
    f *= (1.0 + B * torch.cos(C / l))
    return f

# ============================================================
# Diagnostics Storage
# ============================================================

num_bounces = 15

errors_L2 = []
rho_vals = []
sigma_mins = []
cond_vals = []
energy_drift = []

alpha_current = alpha.clone()
S_gt = S_init.clone()

# ============================================================
# Bounce Loop
# ============================================================

for bounce in range(num_bounces):

    f_lambda = oscillatory_operator(lam)

    # --- Build operator ---
    B = basis.m_basisRaw
    dx = domain.m_delta / basis.m_sigma

    weighted_basis = B * f_lambda.unsqueeze(0)
    O_raw = (weighted_basis @ B.T) * dx
    O = torch.linalg.solve(basis.m_gram, O_raw)

    # --- Whiten ---
    L = basis.m_L
    LinvT = basis.m_LinvT
    O_tilde = L.T @ O @ LinvT

    # --- Operator diagnostics ---
    svals = torch.linalg.svdvals(O_tilde)
    sigma_min = svals.min().item()
    sigma_mins.append(sigma_min)

    cond_vals.append((svals.max() / svals.min()).item())

    eigvals = torch.linalg.eigvals(O_tilde)
    rho_vals.append(torch.max(torch.abs(eigvals)).item())

    # --- Apply ---
    alpha_current = O_tilde @ alpha_current
    S_rec = basis.reconstructWhitened(alpha_current)

    # --- Ground truth ---
    S_gt *= f_lambda

    # --- Normalize ---
    S_rec /= domain.integrate(S_rec)
    S_gt  /= domain.integrate(S_gt)

    # --- L2 ---
    L2 = torch.sqrt(
        domain.integrate((S_gt - S_rec) ** 2)
    ).item()

    errors_L2.append(L2)

    # --- Energy drift ---
    E_gt = domain.integrate(S_gt).item()
    E_op = domain.integrate(S_rec).item()
    energy_drift.append(abs(E_gt - E_op))

# ============================================================
# PLOTS
# ============================================================

x = np.arange(1, num_bounces + 1)

# --- L2 ---
engine1 = PlotEngine(figsize=(8,5))
engine1.addLine(x, np.array(errors_L2), linewidth=2.5, label="L2 Error")
engine1.setTitle("Extreme Oscillation — L2 Error")
engine1.setLabels("Bounce", "L2")
engine1.addLegend()
engine1.show()

# --- Spectral Radius ---
engine2 = PlotEngine(figsize=(8,5))
engine2.addLine(x, np.array(rho_vals), linewidth=2.5, label="Spectral Radius")
engine2.addHorizontalLine(1.0, label="ρ = 1")
engine2.setTitle("Spectral Radius vs Bounce")
engine2.setLabels("Bounce", "ρ(Õ)")
engine2.addLegend()
engine2.show()

# --- σ_min ---
engine3 = PlotEngine(figsize=(8,5))
engine3.addLine(x, np.array(sigma_mins), linewidth=2.5, label="σ_min")
engine3.m_axes.set_yscale("log")
engine3.setTitle("Minimum Singular Value")
engine3.setLabels("Bounce", "σ_min")
engine3.addLegend()
engine3.show()

# --- Condition number ---
engine4 = PlotEngine(figsize=(8,5))
engine4.addLine(x, np.array(cond_vals), linewidth=2.5, label="cond(Õ)")
engine4.m_axes.set_yscale("log")
engine4.setTitle("Operator Conditioning")
engine4.setLabels("Bounce", "Condition Number")
engine4.addLegend()
engine4.show()

# --- Energy Drift ---
engine5 = PlotEngine(figsize=(8,5))
engine5.addLine(x, np.array(energy_drift), linewidth=2.5, label="Energy Drift")
engine5.setTitle("Energy Drift vs Bounce")
engine5.setLabels("Bounce", "Absolute Energy Error")
engine5.addLegend()
engine5.show()

# --- Final Spectrum ---
engine6 = PlotEngine(figsize=(10,4))
engine6.addLine(
    lam.detach().cpu().numpy(),
    S_gt.detach().cpu().numpy(),
    linewidth=2.5,
    label="Ground Truth"
)
engine6.addLine(
    lam.detach().cpu().numpy(),
    S_rec.detach().cpu().numpy(),
    linestyle="--",
    linewidth=2.5,
    label="Projected"
)
engine6.setTitle("Final Spectrum After Extreme Oscillations")
engine6.setLabels("Wavelength (nm)", "Power")
engine6.addLegend()
engine6.show()
