import torch
import numpy as np

from core.SpectralDomain import SpectralDomain
from core.GhgsfMultiLobeBasis import GHGSFMultiLobeBasis
from plotting.Plot import PlotEngine

# ============================================================
# Domain
# ============================================================

device = torch.device("cuda")
dtype = torch.float64

domain = SpectralDomain(
    lambdaMin=400.0,
    lambdaMax=700.0,
    numSamples=2048,
    device=device,
    dtype=dtype
)

lam = domain.m_lambda
torch.set_default_dtype(dtype)

# ============================================================
# Initial Spectrum
# ============================================================

def initial_spectrum(l):
    return (
        0.8 * torch.exp(-0.5 * ((l - 460.0) / 60.0)**2) +
        0.9 * torch.exp(-0.5 * ((l - 560.0) / 70.0)**2)
    )

S_init = initial_spectrum(lam)
S_init /= domain.integrate(S_init)

# ============================================================
# Random Operators
# ============================================================

def random_smooth_spectrum(l, rng):
    spectrum = torch.zeros_like(l)

    for _ in range(rng.integers(2, 5)):
        center = rng.uniform(410.0, 690.0)
        width  = rng.uniform(10.0, 40.0)
        amp    = rng.uniform(0.3, 1.2)

        spectrum += amp * torch.exp(
            -0.5 * ((l - center) / width)**2
        )

    return spectrum

def iridescence_function(l, delta, A=0.4):
    return 1.0 + A * torch.cos(2.0 * torch.pi * delta / l)

# ============================================================
# GHGSF Basis (Stable Sweet Spot)
# ============================================================

sigma = 12
lobes = torch.linspace(420.0, 680.0, 8).tolist()
order = 6

basis = GHGSFMultiLobeBasis(
    domain=domain,
    centers=lobes,
    sigma=sigma,
    order=order
)

cond = torch.linalg.cond(basis.m_gram).item()
print("Gram condition number:", cond)

alpha_init = basis.project(S_init)

# ============================================================
# Correct Galerkin Operator
# ============================================================

def build_operator(f_lambda):

    B = basis.m_basisRaw           # [M, L]
    w = basis.m_domain.m_weights   # [L]

    weighted_basis = B * f_lambda.unsqueeze(0)
    Bw = B * w

    O_raw = Bw @ weighted_basis.T

    return basis.m_gramInv @ O_raw

# ============================================================
# Multi-Bounce Sweep
# ============================================================

rng = np.random.default_rng(42)

max_bounces = 20
errors_L2 = []

S_gt = S_init.clone()
alpha = alpha_init.clone()

for bounce in range(max_bounces):

    sigma_s = random_smooth_spectrum(lam, rng)
    refl    = random_smooth_spectrum(lam, rng)
    delta   = rng.uniform(600.0, 2200.0)

    f_lambda = torch.exp(-sigma_s * rng.uniform(0.1, 1.5))
    f_lambda *= refl
    f_lambda *= iridescence_function(lam, delta)

    O = build_operator(f_lambda)

    # Ground truth update
    S_gt = S_gt * f_lambda

    # Operator update
    alpha = O @ alpha

    S_op = basis.reconstruct(alpha)

    # Normalize
    S_gt_norm = S_gt / domain.integrate(S_gt)
    S_op_norm = S_op / domain.integrate(S_op)

    L2 = torch.sqrt(
        domain.integrate((S_gt_norm - S_op_norm)**2)
    ).item()

    errors_L2.append(L2)

# ============================================================
# Plot 1: Error Growth
# ============================================================

engine_err = PlotEngine(figsize=(8, 5))

engine_err.addLine(
    np.arange(1, max_bounces + 1),
    np.array(errors_L2),
    linewidth=2.2,
    label="L2 Error"
)

engine_err.setTitle(
    f"BsSPT Multi-Bounce Error Growth\nGram cond = {cond:.2e}"
)

engine_err.setLabels("Bounce Depth", "L2 Error")
engine_err.addLegend()

engine_err.show()

# ============================================================
# Plot 2: Final Spectrum Comparison
# ============================================================

engine_spec = PlotEngine(figsize=(10, 4))

lam_cpu = lam.detach().cpu().numpy()

engine_spec.addLine(
    lam_cpu,
    S_gt_norm.detach().cpu().numpy(),
    linewidth=2.5,
    label="Ground Truth"
)

engine_spec.addLine(
    lam_cpu,
    S_op_norm.detach().cpu().numpy(),
    linestyle="--",
    linewidth=2.5,
    label="BsSPT"
)

engine_spec.setTitle("Final Spectrum After Multi-Bounce Transport")
engine_spec.setLabels("Wavelength (nm)", "Power")
engine_spec.addLegend()

engine_spec.show()