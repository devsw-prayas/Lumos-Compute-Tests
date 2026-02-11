import torch
from torch import Tensor

from core.GhgsfMultiLobeBasis import GHGSFMultiLobeBasis

class SpectralState:

    def __init__(self, basis: GHGSFMultiLobeBasis, coeffs: Tensor):
        self.m_basis  = basis
        self.m_coeffs = coeffs.clone()

    def norm(self):
        return torch.linalg.norm(self.m_coeffs)

    def apply(self, operator):
        operator.apply(self)