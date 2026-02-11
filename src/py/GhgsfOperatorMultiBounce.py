import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite, factorial

# ------------------------------------------------------------
# Dark Mode Setup
# ------------------------------------------------------------
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#0f1116",
    "axes.facecolor": "#0f1116",
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
    "legend.frameon": False,
    "grid.alpha": 0.2,
})

# ------------------------------------------------------------
# Wavelength grid
# ------------------------------------------------------------
lam = np.linspace(400.0, 700.0, 4096)
dl = lam[1] - lam[0]

# ------------------------------------------------------------
# Initial spectrum
# ------------------------------------------------------------
def initial_spectrum(l):
    return (
        0.8 * np.exp(-0.5 * ((l - 460.0) / 60.0)**2) +
        0.9 * np.exp(-0.5 * ((l - 560.0) / 70.0)**2)
    )

S_init = initial_spectrum(lam)
S_init /= np.trapezoid(S_init, lam)

# ------------------------------------------------------------
# Random smooth material
# ------------------------------------------------------------
def random_smooth_spectrum(l, rng):
    spectrum = np.zeros_like(l)
    for _ in range(rng.integers(2, 5)):
        center = rng.uniform(410, 690)
        width = rng.uniform(10, 40)
        amp = rng.uniform(0.3, 1.2)
        spectrum += amp * np.exp(-0.5 * ((l - center) / width)**2)
    return spectrum

def iridescence_function(l, delta, A=0.4):
    return 1.0 + A * np.cos(2.0 * np.pi * delta / l)

# ------------------------------------------------------------
# GHGSF Basis: 8 × 6, σ = 12
# ------------------------------------------------------------
sigma_lobe = 12.0
lobes = np.linspace(405, 695, 8)
modes_per_lobe = 6

basis = []
dx = dl / sigma_lobe

for center in lobes:
    x = (lam - center) / sigma_lobe
    for n in range(modes_per_lobe):
        Hn = hermite(n)
        norm = np.sqrt((2.0**n) * factorial(n) * np.sqrt(np.pi))
        phi = (Hn(x) * np.exp(-0.5 * x**2)) / norm
        basis.append(phi)

basis = np.array(basis)
K = basis.shape[0]
basis_T = basis.T

# Gram matrix
G = basis @ basis_T * dx
G_inv = np.linalg.inv(G)

print("Gram condition number:", np.linalg.cond(G))

# Project initial spectrum
b_init = basis @ S_init * dx
alpha_init = G_inv @ b_init

# Fast operator builder (BLAS)
def build_operator_fast(f_lambda):
    weighted_basis = basis * f_lambda
    O_raw = weighted_basis @ basis_T * dx
    return G_inv @ O_raw

# ------------------------------------------------------------
# Ultra High Bounce Test
# ------------------------------------------------------------
rng = np.random.default_rng(42)

num_paths = 32
max_bounces = 15

errors_L2 = []
bounce_levels = []

for bounce_depth in range(1, max_bounces + 1):

    accum_gt = np.zeros_like(lam)
    accum_op = np.zeros_like(lam)

    for p in range(num_paths):

        # Ground truth spectrum
        S_gt = S_init.copy()

        # Operator coefficient
        alpha = alpha_init.copy()

        for _ in range(bounce_depth):

            sigma = random_smooth_spectrum(lam, rng)
            refl = random_smooth_spectrum(lam, rng)
            delta = rng.uniform(600.0, 2200.0)

            f_lambda = np.exp(-sigma * rng.uniform(0.1, 1.5))
            f_lambda *= refl
            f_lambda *= iridescence_function(lam, delta)

            # Ground truth update
            S_gt *= f_lambda

            # Operator update
            O = build_operator_fast(f_lambda)
            alpha = O @ alpha  # GEMV

        accum_gt += S_gt
        accum_op += basis_T @ alpha  # reconstruct once

    accum_gt /= num_paths
    accum_op /= num_paths

    accum_gt /= np.trapezoid(accum_gt, lam)
    accum_op /= np.trapezoid(accum_op, lam)

    L2 = np.sqrt(np.trapezoid((accum_gt - accum_op)**2, lam))

    errors_L2.append(L2)
    bounce_levels.append(bounce_depth)

# ------------------------------------------------------------
# Plot 1: Error Growth
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(bounce_levels, errors_L2, marker='o')
plt.xlabel("Bounce Depth")
plt.ylabel("L2 Error")
plt.title("Operator-Space Spectral Transport Error Growth")
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# Plot 2: Final Spectrum Comparison
# ------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(lam, accum_gt, linewidth=2, label="Ground Truth")
plt.plot(lam, accum_op, '--', linewidth=2, label="Operator BsSPT")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Power")
plt.title("Final Spectrum After Ultra High Bounces")
plt.legend()
plt.grid(True)
plt.show()
