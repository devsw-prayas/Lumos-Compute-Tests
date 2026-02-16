import torch
import math
import matplotlib.pyplot as plt

from core.Dispersion import DispersionOperator
from core.SpectralDomain import SpectralDomain
from core.GhgsfMultiLobeBasis import GHGSFMultiLobeBasis
from core.SpectralState import SpectralState

torch.set_default_dtype(torch.float64)

# -----------------------------
# Spectral domain
# -----------------------------
domain = SpectralDomain(
    lambdaMin=400.0,
    lambdaMax=700.0,
    numSamples=2000,
    device=torch.device("cpu"),
    dtype=torch.float64
)

lbd = domain.m_lambda

# -----------------------------
# GHGSF Basis
# -----------------------------
centers = [430.0, 510.0, 590.0, 670.0]
sigma = 20
order = 6

basis = GHGSFMultiLobeBasis(domain, centers, sigma, order)

# -----------------------------
# Dispersion setup
# -----------------------------
theta_i = torch.tensor(math.radians(45.0))
theta_o = torch.tensor(math.radians(30.0))
sigma_theta = math.radians(0.15)

# -----------------------------
# Continuous dispersion operator
# -----------------------------
eta_i = 1.0
eta = 1.5 + 0.004 / (lbd**2)

sin_theta_t = (eta_i / eta) * torch.sin(theta_i)
theta_t = torch.asin(sin_theta_t)

cos_i = torch.cos(theta_i)
cos_t = torch.sqrt(1.0 - sin_theta_t**2)

rs = ((eta_i * cos_i - eta * cos_t) /
      (eta_i * cos_i + eta * cos_t))**2
rp = ((eta * cos_i - eta_i * cos_t) /
      (eta * cos_i + eta_i * cos_t))**2

R = 0.5 * (rs + rp)
T_lambda = 1.0 - R

selector = torch.exp(
    -(theta_o - theta_t)**2
    / (2.0 * sigma_theta**2)
)

pdf_theta = torch.sum(T_lambda * selector * domain.m_weights)

T = (T_lambda * selector) / pdf_theta

# -----------------------------
# Initial spectrum
# -----------------------------
initial_spectrum = torch.ones_like(lbd)

# Continuous result
continuous = initial_spectrum * T

# -----------------------------
# Basis result
# -----------------------------
coeffs = basis.project(initial_spectrum)
state = SpectralState(basis, coeffs)

disp_op = DispersionOperator(
    basis,
    theta_i,
    theta_o,
    sigma_theta
)

state.apply(disp_op)
projected = basis.reconstruct(state.m_coeffs)

# -----------------------------
# Error
# -----------------------------
error = torch.linalg.norm(projected - continuous) \
        / torch.linalg.norm(continuous)

print("Relative L2 Error:", error.item())

# -----------------------------
# Plots
# -----------------------------
plt.figure()
plt.plot(lbd.numpy(), continuous.numpy(), label="Continuous")
plt.plot(lbd.numpy(), projected.numpy(), label="GHGSF")
plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Spectral Weight")
plt.title("Dispersion Validation")
plt.show()
