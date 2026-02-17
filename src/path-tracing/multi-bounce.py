import torch
import numpy as np

from core.SpectralDomain import SpectralDomain
from core.GhgsfMultiLobeBasis import GHGSFMultiLobeBasis
from plotting.Plot import PlotEngine
import prrequisite

# ============================================================
# Domain
# ============================================================

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_default_dtype(torch.float32)
torch.set_default_device(torch.device("cuda"))

dtype = torch.get_default_dtype()
device = torch.get_default_device()

print(dtype)
print(device)

domain = SpectralDomain(
    lambdaMin=400.0,
    lambdaMax=700.0,
    numSamples=4096,
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

sigma = 8.78
lobes = torch.linspace(405.0, 695.0, 8).tolist()
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
    B = basis.m_basisRaw
    dl = basis.m_domain.m_delta
    dx = dl / basis.m_sigma

    weighted_basis = B * f_lambda.unsqueeze(0)

    O_raw = (weighted_basis @ B.T) * dx

    return basis.m_gramInv @ O_raw

# ============================================================
# Ultra High Bounce Test (PURE TORCH)
# ============================================================

gen = torch.Generator(device=device)
gen.manual_seed(42)

num_paths   = 64
max_bounces = 15

errors_L2 = []
bounce_levels = []

for bounce_depth in range(1, max_bounces + 1):

    accum_gt = torch.zeros_like(lam)
    accum_op = torch.zeros_like(lam)

    for p in range(num_paths):

        S_gt = S_init.clone()
        alpha = alpha_init.clone()

        for _ in range(bounce_depth):

            # ----- random smooth absorption -----
            sigma_s = torch.zeros_like(lam)

            for _ in range(torch.randint(2, 5, (1,), generator=gen, device=device).item()):
                center = 410.0 + 280.0 * torch.rand(1, generator=gen, device=device)
                width  = 10.0 + 30.0 * torch.rand(1, generator=gen, device=device)
                amp    = 0.3 + 0.9 * torch.rand(1, generator=gen, device=device)

                sigma_s += amp * torch.exp(
                    -0.5 * ((lam - center) / width)**2
                )

            # ----- random reflectance -----
            refl = torch.zeros_like(lam)

            for _ in range(torch.randint(2, 5, (1,), generator=gen, device=device).item()):
                center = 410.0 + 280.0 * torch.rand(1, generator=gen, device=device)
                width  = 10.0 + 30.0 * torch.rand(1, generator=gen, device=device)
                amp    = 0.3 + 0.9 * torch.rand(1, generator=gen, device=device)

                refl += amp * torch.exp(
                    -0.5 * ((lam - center) / width)**2
                )

            delta = 600.0 + 1600.0 * torch.rand(1, generator=gen, device=device)

            f_lambda = torch.exp(-sigma_s * (0.1 + 1.4 * torch.rand(1, generator=gen, device=device)))
            f_lambda *= refl
            f_lambda *= (1.0 + 0.4 * torch.cos(2.0 * torch.pi * delta / lam))

            # ----- ground truth update -----
            S_gt = S_gt * f_lambda

            # ----- operator update -----
            O = build_operator(f_lambda)
            alpha = O @ alpha

        accum_gt += S_gt
        accum_op += basis.reconstruct(alpha)

    # ----- average paths -----
    accum_gt /= num_paths
    accum_op /= num_paths

    # ----- normalize -----
    accum_gt /= domain.integrate(accum_gt)
    accum_op /= domain.integrate(accum_op)

    # ----- L2 error -----
    L2 = torch.sqrt(
        domain.integrate((accum_gt - accum_op) ** 2)
    ).item()

    errors_L2.append(L2)
    bounce_levels.append(bounce_depth)


# ============================================================
# Plot 1: Error Growth
# ============================================================

engine_err = PlotEngine(figsize=(8, 5))

engine_err.addLine(
    torch.tensor(bounce_levels).cpu().numpy(),
    torch.tensor(errors_L2).cpu().numpy(),
    linewidth=2.2,
    label="L2 Error"
)

engine_err.setTitle("Operator-Space Spectral Transport Error Growth")
engine_err.setLabels("Bounce Depth", "L2 Error")
engine_err.addLegend()
engine_err.show()

engine_spec = PlotEngine(figsize=(10, 4))

engine_spec.addLine(
    lam.detach().cpu().numpy(),
    accum_gt.detach().cpu().numpy(),
    linewidth=2.5,
    label="Ground Truth"
)

engine_spec.addLine(
    lam.detach().cpu().numpy(),
    accum_op.detach().cpu().numpy(),
    linestyle="--",
    linewidth=2.5,
    label="Operator BsSPT"
)

engine_spec.setTitle("Final Spectrum After Ultra High Bounces")
engine_spec.setLabels("Wavelength (nm)", "Power")
engine_spec.addLegend()
engine_spec.show()
