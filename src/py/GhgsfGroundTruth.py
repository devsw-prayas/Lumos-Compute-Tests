import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite, factorial

# ------------------------------------------------------------
# High-resolution wavelength grid (Ground Truth)
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
# Absorption
# ------------------------------------------------------------
def absorption_sigma(l):
    return (
        0.8 * np.exp(-0.5 * ((l - 430.0) / 10.0)**2) +
        1.1 * np.exp(-0.5 * ((l - 520.0) / 18.0)**2) +
        0.6 * np.exp(-0.5 * ((l - 610.0) / 14.0)**2)
    )

sigma = absorption_sigma(lam)

# ------------------------------------------------------------
# Iridescence
# ------------------------------------------------------------
def iridescence(S, l, delta, A=0.4):
    phase = 2.0 * np.pi * delta / l
    return np.clip(S * (1.0 + A * np.cos(phase)), 0.0, None)

# ------------------------------------------------------------
# GHGSF Basis Construction
# ------------------------------------------------------------
sigma_lobe = 12.0
lobes = np.linspace(410, 690, 8)
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

# ------------------------------------------------------------
# Mutation Comparison
# ------------------------------------------------------------
rng = np.random.default_rng(42)

num_paths = 400
max_bounces = 6

accum_gt = np.zeros_like(lam)
accum_bsspt = np.zeros_like(lam)

for p in range(num_paths):

    S_gt = S_init.copy()
    S_bsspt = S_init.copy()

    num_bounces = rng.integers(1, max_bounces + 1)

    for _ in range(num_bounces):

        d = rng.uniform(0.2, 1.2)
        delta = rng.uniform(600.0, 2200.0)

        # --- Ground Truth ---
        S_gt *= np.exp(-sigma * d)
        S_gt = iridescence(S_gt, lam, delta)

        # --- BsSPT ---
        S_bsspt *= np.exp(-sigma * d)
        S_bsspt = iridescence(S_bsspt, lam, delta)
        S_bsspt = project_and_reconstruct(S_bsspt)

    accum_gt += S_gt
    accum_bsspt += S_bsspt

# Normalize
accum_gt /= num_paths
accum_bsspt /= num_paths

accum_gt /= np.trapezoid(accum_gt, lam)
accum_bsspt /= np.trapezoid(accum_bsspt, lam)

# ------------------------------------------------------------
# Error Metrics
# ------------------------------------------------------------
L2_error = np.sqrt(np.trapezoid((accum_gt - accum_bsspt)**2, lam))
Linf_error = np.max(np.abs(accum_gt - accum_bsspt))
energy_diff = np.abs(
    np.trapezoid(accum_gt, lam) - np.trapezoid(accum_bsspt, lam)
)

print("L2 Error:", L2_error)
print("Linf Error:", Linf_error)
print("Energy Difference:", energy_diff)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(lam, accum_gt, label="Ground Truth", linewidth=2)
plt.plot(lam, accum_bsspt, '--', label="BsSPT", linewidth=2)
plt.title("Ground Truth vs BsSPT Multi-Bounce Spectral Transport")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Power")
plt.legend()
plt.grid(True)
plt.show()
