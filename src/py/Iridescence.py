import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite, factorial

# ------------------------------------------------------------
# Dark mode
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
lam = np.linspace(400.0, 700.0, 256)
dl = lam[1] - lam[0]

# ------------------------------------------------------------
# Initial spectrum (white-ish)
# ------------------------------------------------------------
def initial_spectrum(l):
    return (
        0.8 * np.exp(-0.5 * ((l - 460.0) / 60.0)**2) +
        0.9 * np.exp(-0.5 * ((l - 560.0) / 70.0)**2)
    )

S_init = initial_spectrum(lam)
S_init /= np.trapezoid(S_init, lam)

# ------------------------------------------------------------
# Absorption σ(λ)
# ------------------------------------------------------------
def absorption_sigma(l):
    return (
        0.8 * np.exp(-0.5 * ((l - 430.0) / 10.0)**2) +
        1.1 * np.exp(-0.5 * ((l - 520.0) / 18.0)**2) +
        0.6 * np.exp(-0.5 * ((l - 610.0) / 14.0)**2)
    )

sigma = absorption_sigma(lam)

# ------------------------------------------------------------
# Iridescence mutation
# ------------------------------------------------------------
def iridescence(S, l, delta, A=0.4):
    phase = 2.0 * np.pi * delta / l
    return np.clip(S * (1.0 + A * np.cos(phase)), 0.0, None)

# ------------------------------------------------------------
# GHGSF construction
# ------------------------------------------------------------
sigma_lobe = 8.0
dx = dl / sigma_lobe

lobes = np.linspace(410, 690, 8)
modes_per_lobe = 6

basis = []
for center in lobes:
    x = (lam - center) / sigma_lobe
    for n in range(modes_per_lobe):
        Hn = hermite(n)
        norm = np.sqrt((2.0**n) * factorial(n) * np.sqrt(np.pi))
        phi = (Hn(x) * np.exp(-0.5 * x**2)) / norm
        basis.append(phi)

basis = np.array(basis)
K = basis.shape[0]

G = np.zeros((K, K))
for i in range(K):
    for j in range(K):
        G[i, j] = np.sum(basis[i] * basis[j]) * dx

def project_and_reconstruct(S):
    b = np.array([np.sum(S * phi) * dx for phi in basis])
    alpha = np.linalg.solve(G, b)
    return np.sum(alpha[:, None] * basis, axis=0)

# Pre-project absorption
sigma_hat = project_and_reconstruct(sigma)

# ------------------------------------------------------------
# Mutation run
# ------------------------------------------------------------
rng = np.random.default_rng(7)

num_paths   = 400      # accumulation size
max_bounces = 6

accum = np.zeros_like(lam)

for p in range(num_paths):
    S = S_init.copy()

    num_bounces = rng.integers(1, max_bounces + 1)

    for _ in range(num_bounces):
        # Random Beer–Lambert distance
        d = rng.uniform(0.2, 1.2)
        S *= np.exp(-sigma_hat * d)

        # Random iridescence mutation
        delta = rng.uniform(600.0, 2200.0)
        S = iridescence(S, lam, delta)

        # Reproject to GHGSF space
        S = project_and_reconstruct(S)

    accum += S

# Normalize accumulation
accum /= num_paths
accum /= np.trapezoid(accum, lam)

# ------------------------------------------------------------
# Plot result
# ------------------------------------------------------------
plt.figure(figsize=(11, 4))
plt.plot(lam, accum, linewidth=2, label="Accumulated spectrum")
plt.plot(lam, S_init, "--", linewidth=2, label="Initial spectrum")
plt.title(
    f"Mutation Run — Multi-Bounce Accumulation\n"
    f"{num_paths} paths, up to {max_bounces} bounces"
)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Power")
plt.grid(True)
plt.legend()
plt.show()
