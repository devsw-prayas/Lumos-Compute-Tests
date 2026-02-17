import torch
import numpy as np

from core.SpectralDomain import SpectralDomain
from core.GhgsfMultiLobeBasis import GHGSFMultiLobeBasis
from plotting.Plot import PlotEngine, MultiPanelEngine

# --------------------------------------------------
# Configuration
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lambdaMin = 380.0
lambdaMax = 780.0
numSamples = 4000  # high resolution for stable Gram

domain = SpectralDomain(
    lambdaMin=lambdaMin,
    lambdaMax=lambdaMax,
    numSamples=numSamples,
    device=device,
    dtype=torch.float64
)

K_values = [4,5,6,7,8]
N_values = [4,5,6,7,8]
sigma_values = [6,7,8,9,10,11,12,13]

results = {}

# --------------------------------------------------
# Sweep
# --------------------------------------------------

for sigma in sigma_values:

    heat = np.zeros((len(K_values), len(N_values)))

    for i, K in enumerate(K_values):
        centers = np.linspace(lambdaMin, lambdaMax, K)

        for j, N in enumerate(N_values):

            basis = GHGSFMultiLobeBasis(
                domain=domain,
                centers=centers.tolist(),
                sigma=sigma,
                order=N
            )

            G = basis.m_gram.detach().cpu()

            # SVD for conditioning
            s = torch.linalg.svdvals(G)
            kappa = (s.max() / s.min()).item()

            heat[i, j] = np.log10(kappa)

    results[sigma] = heat

print("Sweep complete.")

panelEngine = MultiPanelEngine(
    nrows=4,
    ncols=2,
    figsize=(14,18),
    sharex=False
)

panelEngine.setMainTitle("GHGSF Gram Conditioning (log10 κ(G))")

for idx, sigma in enumerate(sigma_values):

    panel = panelEngine.getPanel(idx)

    heat = results[sigma]

    im = panel.m_axes.imshow(
        heat,
        origin='lower',
        aspect='auto'
    )

    panel.setTitle(f"σ = {sigma} nm")
    panel.setLabels("Order N", "Lobes K")

    panel.m_axes.set_xticks(range(len(N_values)))
    panel.m_axes.set_yticks(range(len(K_values)))

    panel.m_axes.set_xticklabels(N_values)
    panel.m_axes.set_yticklabels(K_values)

panelEngine.applyDenseLayout()
panelEngine.show()


plot = PlotEngine(figsize=(8,6))

for (K,N) in [(4,4),(6,6),(8,8)]:

    curve = []

    for sigma in sigma_values:
        heat = results[sigma]
        i = K_values.index(K)
        j = N_values.index(N)
        curve.append(heat[i,j])

    plot.addLine(
        np.array(sigma_values),
        np.array(curve),
        label=f"K={K}, N={N}"
    )

plot.setTitle("Conditioning vs Spectral Overlap")
plot.setLabels("σ (nm)", "log10 κ(G)")
plot.addLegend()
plot.show()

