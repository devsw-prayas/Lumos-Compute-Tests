import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite, factorial

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
# Spectral test cases
# ------------------------------------------------------------
def d65(l):
    return (
        1.0 * np.exp(-0.5 * ((l - 445.0) / 40.0)**2) +
        0.9 * np.exp(-0.5 * ((l - 540.0) / 50.0)**2) +
        0.6 * np.exp(-0.5 * ((l - 610.0) / 60.0)**2)
    )

def leds(l):
    return (
        1.2 * np.exp(-0.5 * ((l - 450.0) / 6.0)**2) +
        1.0 * np.exp(-0.5 * ((l - 530.0) / 5.0)**2) +
        0.9 * np.exp(-0.5 * ((l - 625.0) / 7.0)**2)
    )

def fluorescence(l):
    absorb = np.exp(-0.5 * ((l - 420.0) / 15.0)**2)
    emit   = np.exp(-0.5 * ((l - 580.0) / 20.0)**2)
    return 0.4 * absorb + 1.2 * emit

def reflectance(l):
    return np.clip(
        0.6
        - 0.4 * np.exp(-0.5 * ((l - 500.0) / 30.0)**2)
        - 0.3 * np.exp(-0.5 * ((l - 650.0) / 20.0)**2),
        0.0, 1.0
    )

def dispersion(l):
    base = leds(l)
    phase = 0.002 * (l - 555.0)**2
    return base * np.exp(1j * phase)


cases = {
    "D65": d65(lam),
    "LEDs": leds(lam),
    "Fluorescence": fluorescence(lam),
    "Reflectance": reflectance(lam),
    "Dispersion": dispersion(lam)
}

# Normalize
for k in cases:
    S = cases[k]
    norm = np.trapezoid(np.abs(S), lam) if np.iscomplexobj(S) else np.trapezoid(S, lam)
    cases[k] = S / norm

# ------------------------------------------------------------
# Frame construction + projection
# ------------------------------------------------------------
def build_basis(lobes, modes_per_lobe, sigma):
    basis = []
    for center in lobes:
        x = (lam - center) / sigma
        for n in range(modes_per_lobe):
            Hn = hermite(n)
            norm = np.sqrt((2.0**n) * factorial(n) * np.sqrt(np.pi))
            phi = (Hn(x) * np.exp(-0.5 * x**2)) / norm
            basis.append(phi)
    return np.array(basis)

def gram_matrix(basis, dx):
    K = basis.shape[0]
    G = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            G[i, j] = np.sum(basis[i] * basis[j]) * dx
    return G

def project_and_reconstruct(S, basis, G, dx):
    b = np.array([np.sum(S * phi) * dx for phi in basis])
    alpha = np.linalg.solve(G, b)
    R = np.sum(alpha[:, None] * basis, axis=0)
    return R, alpha

# ------------------------------------------------------------
# Stress sweep
# ------------------------------------------------------------
sigma = 20.0
dx = dl / sigma

configs = [
    (4, np.linspace(430, 670, 4)),
    (6, np.linspace(420, 670, 6)),
    (8, np.linspace(410, 690, 8)),
]

orders = [4, 6, 8]

for modes_per_lobe in orders:
    for n_lobes, lobes in configs:

        basis = build_basis(lobes, modes_per_lobe, sigma)
        G = gram_matrix(basis, dx)
        cond = np.linalg.cond(G)

        fig, axes = plt.subplots(len(cases), 1, figsize=(11, 16))

        for ax, (name, S) in zip(axes, cases.items()):
            R, _ = project_and_reconstruct(S, basis, G, dx)

            if np.iscomplexobj(S):
                ax.plot(lam, np.abs(S), linewidth=2, label=f"{name} |S|")
                ax.plot(lam, np.abs(R), "--", linewidth=2, label="Reconstruction |S|")
            else:
                ax.plot(lam, S, linewidth=2, label=name)
                ax.plot(lam, R, "--", linewidth=2, label="Reconstruction")

            err = np.sqrt(np.trapezoid(np.abs(S - R)**2, lam))
            ax.set_ylabel("Power")
            ax.set_title(f"L² error = {err:.2e}")
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel("Wavelength (nm)")

        fig.suptitle(
            f"GHGSF Stress Test — {n_lobes} lobes × {modes_per_lobe} modes\n"
            f"σ = {sigma} nm   |   Gram cond = {cond:.2e}\n"
            r"Unitary evolution:  $\frac{\partial \Phi}{\partial \theta} = -i\,\mathcal{H}\,\Phi$",
            y=0.95,
            fontsize=18
        )

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()
