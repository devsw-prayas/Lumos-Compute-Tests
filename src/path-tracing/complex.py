import torch
import numpy as np
from plotting.Plot import PlotEngine

torch.set_default_dtype(torch.float64)
torch.set_default_device(torch.device("cuda"))

device = torch.get_default_device()


# ============================================================
# GLOBAL SETUP
# ============================================================

dtype  = torch.get_default_dtype()

# ============================================================
# Spectral Domain
# ============================================================

class SpectralDomain:

    def __init__(self, lambdaMin, lambdaMax, numSamples):

        self.lambdaMin = lambdaMin
        self.lambdaMax = lambdaMax
        self.numSamples = numSamples

        self.m_lambda = torch.linspace(
            lambdaMin, lambdaMax, numSamples,
            device=device,
            dtype=dtype
        )

        self.m_delta = (lambdaMax - lambdaMin) / (numSamples - 1)

        self.m_weights = torch.ones_like(self.m_lambda) * self.m_delta

    def integrate(self, f):
        return torch.sum(f * self.m_weights)

    def innerProduct(self, f, g):
        return self.integrate(torch.conj(f) * g)


domain = SpectralDomain(400.0, 700.0, 4096)
lam = domain.m_lambda

# ============================================================
# Complex GHGSF Basis
# ============================================================

class GHGSFComplex:

    def __init__(self, domain, centers, sigma, order):

        self.Linv = None
        self.L = None
        self.Ginv = None
        self.G = None
        self.B = None
        self.LH = None
        self.LinvH = None
        self.domain = domain
        self.sigma = sigma
        self.order = order

        self.centers = torch.tensor(
            centers,
            device=device,
            dtype=torch.float64
        )

        self.M = len(centers) * order

        self.build_basis()
        self.build_gram()
        self.build_whitening()

    def hermite(self, n, x):
        if n == 0:
            return torch.ones_like(x)
        elif n == 1:
            return 2*x
        H0 = torch.ones_like(x)
        H1 = 2*x
        for k in range(2, n+1):
            Hk = 2*x*H1 - 2*(k-1)*H0
            H0, H1 = H1, Hk
        return H1

    def build_basis(self):

        lam = self.domain.m_lambda
        sigma = self.sigma

        basis = []

        for mu in self.centers:

            x = (lam - mu) / sigma

            for n in range(self.order):

                Hn = self.hermite(n, x)

                norm = torch.sqrt(
                    (2.0**n)
                    * torch.exp(torch.lgamma(torch.tensor(n+1.0)))
                    * torch.sqrt(torch.tensor(np.pi))
                )

                gaussian = torch.exp(-0.5 * x**2)

                # Complex carrier
                omega = 0.02
                phase = torch.exp(1j * omega * (lam - mu))

                phi = (Hn * gaussian / norm) * phase

                basis.append(phi)

        self.B = torch.stack(basis).to(torch.complex128)

    def build_gram(self):

        w = self.domain.m_weights
        weighted = self.B * w

        self.G = weighted @ self.B.conj().T

        I = torch.eye(self.G.shape[0], device=device, dtype=torch.complex128)
        self.Ginv = torch.linalg.solve(self.G, I)

    def build_whitening(self):

        self.L = torch.linalg.cholesky(self.G)

        I = torch.eye(self.G.shape[0], device=device, dtype=torch.complex128)
        self.Linv = torch.linalg.solve(self.L, I)

        self.LH = self.L.conj().T
        self.LinvH = self.Linv.conj().T

    def project(self, S):

        w = self.domain.m_weights
        b = (self.B.conj() * w) @ S

        return torch.linalg.solve(self.G, b)

    def reconstruct(self, alpha):
        return alpha @ self.B

    def projectWhitened(self, S):
        alpha = self.project(S)
        return self.LH @ alpha

    def reconstructWhitened(self, alpha_tilde):
        alpha = self.LinvH @ alpha_tilde
        return self.reconstruct(alpha)



# ============================================================
# Instantiate Basis
# ============================================================

sigma = 8.0
lobes = torch.linspace(405, 695, 8).tolist()
order = 6

basis = GHGSFComplex(domain, lobes, sigma, order)

print("Gram condition number:",
      torch.linalg.cond(basis.G).item())


# ============================================================
# Initial Complex Amplitude
# ============================================================

def initial_amplitude(l):
    return torch.exp(-0.5*((l-550)/60)**2) * torch.exp(1j * 0.01 * l)

phi = initial_amplitude(lam).to(torch.complex128)

phi /= torch.sqrt(domain.integrate(torch.abs(phi)**2))

alpha = basis.projectWhitened(phi)


# ============================================================
# Pure Phase Operator (Unitary Test)
# ============================================================

def phase_operator(l):
    theta = 0.03*l + 8000.0/l
    return torch.exp(1j * theta)


# ============================================================
# Diagnostics
# ============================================================

num_bounces = 15

energy_vals = []
rho_vals = []
L2_vals = []

phi_gt = phi.clone()
alpha_current = alpha.clone()
phi_rec = torch.zeros_like(lam, dtype=torch.complex128)

for b in range(num_bounces):

    f = phase_operator(lam)

    w = domain.m_weights
    weighted = basis.B * f.unsqueeze(0) * w

    O_raw = weighted @ basis.B.conj().T
    O = torch.linalg.solve(basis.G, O_raw)

    O_tilde = basis.LH @ O @ basis.LinvH

    alpha_current = O_tilde @ alpha_current
    phi_rec = basis.reconstructWhitened(alpha_current)

    phi_gt *= f

    energy = domain.integrate(torch.abs(phi_rec)**2).real.item()
    energy_vals.append(energy)

    eigvals = torch.linalg.eigvals(O_tilde)
    rho_vals.append(torch.max(torch.abs(eigvals)).real.item())

    L2 = torch.sqrt(
        domain.integrate(torch.abs(phi_gt - phi_rec)**2)
    ).real.item()

    L2_vals.append(L2)

# ============================================================
# Plots
# ============================================================

phi_gt_abs = torch.abs(phi_gt)
phi_rec_abs = torch.abs(phi_rec)

engine_spec = PlotEngine(figsize=(10, 4))

engine_spec.addLine(
    lam.detach().cpu().numpy(),
    phi_gt_abs.detach().cpu().numpy(),
    linewidth=2.5,
    label="Ground Truth |φ|"
)

engine_spec.addLine(
    lam.detach().cpu().numpy(),
    phi_rec_abs.detach().cpu().numpy(),
    linestyle="--",
    linewidth=2.5,
    label="Projected |φ̂|"
)

engine_spec.setTitle("Final Amplitude Spectrum After Bounces")
engine_spec.setLabels("Wavelength (nm)", "Amplitude")
engine_spec.addLegend()
engine_spec.show()


x = np.arange(1, num_bounces + 1)

engine1 = PlotEngine(figsize=(8,5))
engine1.addLine(x, np.array(energy_vals), label="Energy")
engine1.setTitle("Energy vs Bounce (Unitary Test)")
engine1.setLabels("Bounce", "Energy")
engine1.show()

engine2 = PlotEngine(figsize=(8,5))
engine2.addLine(x, np.array(rho_vals), label="Spectral Radius")
engine2.addHorizontalLine(1.0)
engine2.setTitle("Spectral Radius")
engine2.setLabels("Bounce", "ρ")
engine2.show()

engine3 = PlotEngine(figsize=(8,5))
engine3.addLine(x, np.array(L2_vals), label="L2 Error")
engine3.setTitle("L2 Error vs Bounce")
engine3.setLabels("Bounce", "L2")
engine3.show()
