import random
import numpy as np
import torch

class SandboxRuntime:
    """
    Centralized runtime bootstrap for research sandbox.
    Handles determinism, threading, precision mode, and plotting configuration.
    """

    @staticmethod
    def configureDeterminism(seed: int = 42) -> None:
        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.use_deterministic_algorithms(True)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        print(f"[SandboxRuntime] Deterministic mode enabled (seed={seed})")


    @staticmethod
    def configureThreading(numThreads: int = 8) -> None:
        torch.set_num_threads(numThreads)
        print(f"[SandboxRuntime] Torch threads set to {numThreads}")


    @staticmethod
    def configurePrecisionTF32() -> None:
        """
        Enables TF32 on supported GPUs.
        Default compute dtype remains float32.
        """

        torch.set_float32_matmul_precision("high")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        torch.set_default_dtype(torch.float32)

        print("[SandboxRuntime] TF32 enabled (default float32 compute)")


    @staticmethod
    def bootstrap(
        seed: int = 42,
        numThreads: int = 8
    ) -> None:

        SandboxRuntime.configureDeterminism(seed)
        SandboxRuntime.configureThreading(numThreads)
        SandboxRuntime.configurePrecisionTF32()

        print("[SandboxRuntime] Environment ready 🚀")
