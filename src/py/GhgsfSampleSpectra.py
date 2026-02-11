# Fix mathtext issue by removing \displaystyle and retry plotting

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

lam = np.linspace(400.0, 700.0, 256)
dl = lam[1] - lam[0]

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

for k in cases:
    if np.iscomplexobj(cases[k]):
        cases[k] /= np.trapezoid(np.abs(cases[k]), lam)
    else:
        cases[k] /= np.trapezoid(cases[k], lam)

lobes = np.array([420, 470, 520, 570, 620, 670], dtype=float)
modes_per_lobe = 4
sigma = 20.0

basis = []
for center in lobes:
    x = (lam - center) / sigma
    for n in range(modes_per_lobe):
        Hn = hermite(n)
        norm = np.sqrt((2.0**n) * factorial(n) * np.sqrt(np.pi))
        phi = (Hn(x) * np.exp(-0.5 * x**2)) / norm
        basis.append(phi)

basis = np.array(basis)
dx = dl / sigma
K = basis.shape[0]

G = np.zeros((K, K))
for i in range(K):
    for j in range(K):
        G[i, j] = np.sum(basis[i] * basis[j]) * dx

def project_and_reconstruct(S):
    b = np.array([np.sum(S * phi) * dx for phi in basis])
    alpha = np.linalg.solve(G, b)
    return np.sum(alpha[:, None] * basis, axis=0)

fig, axes = plt.subplots(len(cases), 1, figsize=(11, 16))

for ax, (name, S) in zip(axes, cases.items()):
    R = project_and_reconstruct(S)
    if np.iscomplexobj(S):
        ax.plot(lam, np.abs(S), linewidth=2, label=f"{name} |S|")
        ax.plot(lam, np.abs(R), "--", linewidth=2, label="Reconstruction |S|")
    else:
        ax.plot(lam, S, linewidth=2, label=name)
        ax.plot(lam, R, "--", linewidth=2, label="Reconstruction")
    ax.set_ylabel("Power")
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("Wavelength (nm)")

fig.suptitle(
    "Stable Multi-Lobe GHGSF Frame (Dark Mode)\n"
    "6 lobes × 4 modes, σ = 20 nm, Gram-corrected\n"
    r"Unitary evolution:  $\frac{\partial \Phi}{\partial \theta} = -\,i\,\mathcal{H}\,\Phi$",
    y=0.94,
    fontsize=18
)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()
