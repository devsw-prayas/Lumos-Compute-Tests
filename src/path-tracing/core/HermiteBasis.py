import torch
from torch import Tensor

def hermiteBasis(order: int, x: Tensor) -> Tensor:
    """
    x: [K, L]
    returns:
        H: [K, N, L]
    """

    K, L = x.shape
    N = order

    H = torch.zeros(
        (K, N, L),
        device=x.device,
        dtype=x.dtype
    )

    H[:, 0, :] = 1.0

    if N > 1:
        H[:, 1, :] = 2.0 * x

    for n in range(2, N):
        H[:, n, :] = (
            2.0 * x * H[:, n - 1, :]
            - 2.0 * (n - 1) * H[:, n - 2, :]
        )

    return H

