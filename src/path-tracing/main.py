import torch

# Absolute imports (clean sibling import)
from core.SpectralDomain import SpectralDomain
from core.GhgsfMultiLobeBasis import GHGSFMultiLobeBasis
from core.SpectralState import SpectralState
from core.Absorption import AbsorptionOperator


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1️⃣ Build spectral domain
    domain = SpectralDomain(
        lambdaMin=400.0,
        lambdaMax=700.0,
        numSamples=512,
        device=device
    )

    basis = GHGSFMultiLobeBasis(
        domain=domain,
        centers=[420, 470, 520, 570, 620, 670],
        sigma=20.0,
        order=4
    )

    lbda = domain.m_lambda
    initialSpectrum = torch.exp(
        -0.5 * ((lbda - 520.0) / 25.0) ** 2
    )

    coeffs = basis.project(initialSpectrum)

    state = SpectralState(basis, coeffs)

    print("Initial coefficient norm:", state.norm().item())

    def sigmaA(lbda):
        return 0.02 * torch.exp(
            -0.5 * ((lbda - 550.0) / 40.0) ** 2
        )

    absorption = AbsorptionOperator(
        basis=basis,
        sigmaA=sigmaA,
        distance=1.0
    )

    absorption.apply(state)

    print("Post-absorption norm:", state.norm().item())

    reconstructed = basis.reconstruct(state.m_coeffs)

    print("Reconstruction shape:", reconstructed.shape)


if __name__ == "__main__":
    main()
