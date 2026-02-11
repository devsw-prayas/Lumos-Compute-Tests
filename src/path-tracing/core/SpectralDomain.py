import torch
from torch import Tensor

class SpectralDomain:

    def __init__(
        self,
        lambdaMin: float,
        lambdaMax: float,
        numSamples: int,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float64
    ):
        self.m_device = device
        self.m_dtype  = dtype

        self.m_lambda = torch.linspace(
            lambdaMin,
            lambdaMax,
            numSamples,
            device=device,
            dtype=dtype
        )

        self.m_delta = self.m_lambda[1] - self.m_lambda[0]
        self.m_count = numSamples

        # Trapezoidal weights
        w = torch.ones(numSamples, device=device, dtype=dtype)
        w[0] *= 0.5
        w[-1] *= 0.5
        self.m_weights = w * self.m_delta

    def integrate(self, f: Tensor) -> Tensor:
        return torch.sum(f * self.m_weights)

    def innerProduct(self, f: Tensor, g: Tensor) -> Tensor:
        return self.integrate(f * g)
