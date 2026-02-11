import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite, factorial

# ------------------------------------------------------------
# High-resolution wavelength grid
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
# Random smooth material generator
# ------------------------------------------------------------
def random_smooth_spectrum(l, rng):
    spectrum = np.zeros_like(l)
    for _ in range(rng.integers(2, 5)):
        center = rng.uniform(410, 690)
        width = rng.uniform(10, 40)
        amp = rng.uniform(0.3, 1.2)
        spectrum += amp * np.exp(-0.5 * ((l - center) / width)**2)
    spectrum = np.clip(spectrum, 0.0, None)
    return spectrum / (np.max(spectrum) + 1e-8)

# ------------------------------------------------------------
# Iridescence modulation
# ------------------------------------------------------------
def iridescence(S, l, delta, A=0.4):
    phase = 2.0 * np.pi * delta / l
    return np.clip(S * (1.0 + A * np.cos(phase)), 0.0, None)

# ------------------------------------------------------------
# GHGSF basis: 8 lobes × order 6, σ = 12
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

# Gram matrix
G = np.zeros((K, K))
for i in range(K):
    for j in range(K):
        G[i, j] = np.sum(basis[i] * basis[j]) * dx

G_inv = np.linalg.inv(G)

def project_and_reconstruct(S):
    b = np.array([np.sum(S * phi) * dx for phi in basis])
    alpha = G_inv @ b
    return np.sum(alpha[:, None] * basis, axis=0)

print("Gram condition number:", np.linalg.cond(G))

# ------------------------------------------------------------
# Ultra high-bounce stress
# ------------------------------------------------------------
rng = np.random.default_rng(123)

num_paths = 200
max_bounces = 30

errors_L2 = []
bounce_levels = []

for bounce_depth in range(1, max_bounces + 1):

    accum_gt = np.zeros_like(lam)
    accum_bsspt = np.zeros_like(lam)

    for p in range(num_paths):

        S_gt = S_init.copy()
        S_bsspt = S_init.copy()

        for _ in range(bounce_depth):

            # Random absorption
            sigma = random_smooth_spectrum(lam, rng)
            d = rng.uniform(0.1, 1.5)

            S_gt *= np.exp(-sigma * d)
            S_bsspt *= np.exp(-sigma * d)

            # Random reflectance
            refl = random_smooth_spectrum(lam, rng)
            S_gt *= refl
            S_bsspt *= refl

            # Random iridescence
            delta = rng.uniform(600.0, 2200.0)
            S_gt = iridescence(S_gt, lam, delta)
            S_bsspt = iridescence(S_bsspt, lam, delta)

            # Projection only in BsSPT
            S_bsspt = project_and_reconstruct(S_bsspt)

        accum_gt += S_gt
        accum_bsspt += S_bsspt

    accum_gt /= num_paths
    accum_bsspt /= num_paths

    accum_gt /= np.trapezoid(accum_gt, lam)
    accum_bsspt /= np.trapezoid(accum_bsspt, lam)

    L2 = np.sqrt(np.trapezoid((accum_gt - accum_bsspt)**2, lam))

    errors_L2.append(L2)
    bounce_levels.append(bounce_depth)

# ------------------------------------------------------------
# Plot error growth
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(bounce_levels, errors_L2, marker='o')
plt.xlabel("Bounce Depth")
plt.ylabel("L2 Error vs Ground Truth")
plt.title("Ultra High-Bounce Spectral Transport Stability Test")
plt.grid(True)
plt.show()
    