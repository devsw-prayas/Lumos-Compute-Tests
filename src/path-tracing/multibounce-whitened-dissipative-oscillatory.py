import torch
import numpy as np

from core.SpectralDomain import SpectralDomain
from core.GhgsfMultiLobeBasis import GHGSFMultiLobeBasis
from plotting.Plot import PlotEngine


# ============================================================
# Domain
# ============================================================

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_default_dtype(torch.float64)
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
# GHGSF Basis
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

print("Gram condition number:", torch.linalg.cond(basis.m_gram).item())

# Whitened initial coefficients
alpha_init = basis.projectWhitened(S_init)


# ============================================================
# Whitened Galerkin Operator
# ============================================================

def build_operator_whitened(f_lambda):

    B = basis.m_basisRaw
    dl = basis.m_domain.m_delta
    dx = dl / basis.m_sigma

    weighted_basis = B * f_lambda.unsqueeze(0)
    O_raw = (weighted_basis @ B.T) * dx

    # Unwhitened operator
    O = torch.linalg.solve(basis.m_gram, O_raw)

    # Whitening transform
    LT = basis.m_LT
    LinvT = basis.m_LinvT

    # O_tilde = L^T O L^{-T}
    O_tilde = LT @ O @ LinvT

    return O_tilde


# ============================================================
# Diagnostics Storage
# ============================================================

mean_operator_conds = []
mean_operator_radii = []
mean_sigma_mins = []
mean_energy_errors = []
errors_L2 = []
bounce_levels = []


# ============================================================
# Ultra High Bounce Test
# ============================================================

gen = torch.Generator(device=device)
gen.manual_seed(42)

num_paths   = 64
max_bounces = 15

for bounce_depth in range(1, max_bounces + 1):

    operator_conds = []
    operator_radii = []
    sigma_mins = []
    energy_errors = []

    accum_gt = torch.zeros_like(lam)
    accum_op = torch.zeros_like(lam)

    for p in range(num_paths):

        S_gt = S_init.clone()
        alpha = alpha_init.clone()

        for _ in range(bounce_depth):

            # Random absorption
            sigma_s = torch.zeros_like(lam)

            for _ in range(torch.randint(2, 5, (1,), generator=gen, device=device).item()):
                center = 410.0 + 280.0 * torch.rand(1, generator=gen, device=device)
                width  = 10.0 + 30.0 * torch.rand(1, generator=gen, device=device)
                amp    = 0.3 + 0.9 * torch.rand(1, generator=gen, device=device)

                sigma_s += amp * torch.exp(
                    -0.5 * ((lam - center) / width)**2
                )

            # Random reflectance
            refl = torch.zeros_like(lam)

            for _ in range(torch.randint(2, 5, (1,), generator=gen, device=device).item()):
                center = 410.0 + 280.0 * torch.rand(1, generator=gen, device=device)
                width  = 10.0 + 30.0 * torch.rand(1, generator=gen, device=device)
                amp    = 0.3 + 0.9 * torch.rand(1, generator=gen, device=device)

                refl += amp * torch.exp(
                    -0.5 * ((lam - center) / width)**2
                )

            delta = 200.0 + 4000.0 * torch.rand(1, generator=gen, device=device)

            # Remove absorption completely
            refl = refl / torch.max(refl)  # normalize reflectance to ~1

            f_lambda = refl
            f_lambda *= (1.0 + 0.8  * torch.cos(2.0 * torch.pi * delta / lam))

            # Ground truth
            S_gt = S_gt * f_lambda

            # Whitened operator update
            O_tilde = build_operator_whitened(f_lambda)

            # Diagnostics
            svals = torch.linalg.svdvals(O_tilde)
            condO = (svals.max() / svals.min()).item()
            operator_conds.append(condO)
            sigma_mins.append(svals.min().item())

            eigvals = torch.linalg.eigvals(O_tilde)
            rho = torch.max(torch.abs(eigvals)).item()
            operator_radii.append(rho)

            # Evolve whitened state
            alpha = O_tilde @ alpha

            E_gt = domain.integrate(S_gt).item()
            E_op = domain.integrate(basis.reconstructWhitened(alpha)).item()
            energy_errors.append(abs(E_gt - E_op))

        accum_gt += S_gt
        accum_op += basis.reconstructWhitened(alpha)

    accum_gt /= num_paths
    accum_op /= num_paths

    accum_gt /= domain.integrate(accum_gt)
    accum_op /= domain.integrate(accum_op)

    L2 = torch.sqrt(
        domain.integrate((accum_gt - accum_op) ** 2)
    ).item()

    errors_L2.append(L2)
    bounce_levels.append(bounce_depth)

    mean_operator_conds.append(np.mean(operator_conds))
    mean_operator_radii.append(np.mean(operator_radii))
    mean_sigma_mins.append(np.mean(sigma_mins))
    mean_energy_errors.append(np.mean(energy_errors))


# ============================================================
# Plots
# ============================================================

engine_err = PlotEngine(figsize=(8, 5))
engine_err.addLine(np.array(bounce_levels), np.array(errors_L2), linewidth=2.2, label="L2 Error")
engine_err.setTitle("Whitened Operator-Space Spectral Transport Error Growth")
engine_err.setLabels("Bounce Depth", "L2 Error")
engine_err.addLegend()
engine_err.show()

engine_condO = PlotEngine(figsize=(8, 5))
engine_condO.addLine(np.array(bounce_levels), np.array(mean_operator_conds), linewidth=2.2, label="cond(Õ)")
engine_condO.setTitle("Whitened Operator Conditioning vs Bounce Depth")
engine_condO.setLabels("Bounce Depth", "Condition Number")
engine_condO.addLegend()
engine_condO.m_axes.set_yscale("log")
engine_condO.show()

engine_rho = PlotEngine(figsize=(8, 5))
engine_rho.addLine(np.array(bounce_levels), np.array(mean_operator_radii), linewidth=2.2, label="Spectral Radius")
engine_rho.addHorizontalLine(1.0, label="ρ = 1")
engine_rho.setTitle("Whitened Spectral Radius vs Bounce Depth")
engine_rho.setLabels("Bounce Depth", "ρ(Õ)")
engine_rho.addLegend()
engine_rho.show()

engine_sigma = PlotEngine(figsize=(8, 5))
engine_sigma.addLine(np.array(bounce_levels), np.array(mean_sigma_mins), linewidth=2.2, label="σ_min(Õ)")
engine_sigma.setTitle("Whitened Minimum Singular Value vs Bounce Depth")
engine_sigma.setLabels("Bounce Depth", "σ_min")
engine_sigma.addLegend()
engine_sigma.m_axes.set_yscale("log")
engine_sigma.show()

engine_energy = PlotEngine(figsize=(8, 5))
engine_energy.addLine(np.array(bounce_levels), np.array(mean_energy_errors), linewidth=2.2, label="Energy Drift")
engine_energy.setTitle("Whitened Energy Drift vs Bounce Depth")
engine_energy.setLabels("Bounce Depth", "Absolute Energy Error")
engine_energy.addLegend()
engine_energy.show()
