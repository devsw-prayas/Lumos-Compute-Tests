import torch

from core.SpectralOperator import SpectralOperator


class DispersionOperator(SpectralOperator):

    def __init__(self, basis, theta_i, theta_o, sigma_theta):
        super().__init__(basis)
        self.theta_i = theta_i
        self.theta_o = theta_o
        self.sigma_theta = sigma_theta

    def buildMatrix(self):
        lbd = self.m_basis.m_domain.m_lambda
        B = self.m_basis.m_basisRaw
        G_inv = self.m_basis.m_gramInv

        # --- Same integration metric as Gram ---
        dl = self.m_basis.m_domain.m_delta
        dx = dl / self.m_basis.m_sigma

        # Cauchy IOR
        eta_i = 1.0
        eta = 1.5 + 0.004 / (lbd ** 2)

        # Snell
        sin_theta_t = (eta_i / eta) * torch.sin(self.theta_i)
        theta_t = torch.asin(sin_theta_t)

        # Fresnel transmission (unpolarized)
        cos_i = torch.cos(self.theta_i)
        cos_t = torch.sqrt(1.0 - sin_theta_t ** 2)

        rs = ((eta_i * cos_i - eta * cos_t) / (eta_i * cos_i + eta * cos_t)) ** 2
        rp = ((eta * cos_i - eta_i * cos_t) / (eta * cos_i + eta_i * cos_t)) ** 2
        R = 0.5 * (rs + rp)
        T_lambda = 1.0 - R

        # Gaussian selector
        selector = torch.exp(
            -(self.theta_o - theta_t) ** 2
            / (2.0 * self.sigma_theta ** 2)
        )

        # --- Normalize in physical wavelength space ---
        w = self.m_basis.m_domain.m_weights
        pdf_theta = torch.sum(T_lambda * selector * w)

        T = (T_lambda * selector) / pdf_theta

        # --- Galerkin assembly using dx ---
        weighted = B * (dx * T)
        M_raw = weighted @ B.T

        self.m_matrix = G_inv @ M_raw
