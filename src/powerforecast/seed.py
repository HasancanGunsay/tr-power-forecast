"""Reproducibility helpers.

An experiment you cannot reproduce is an anecdote, not a result. Seeding is
the cheapest step toward reproducibility, so it belongs in the template rather
than being rediscovered in every project.
"""

from __future__ import annotations

import os
import random

from powerforecast.config import RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED, *, deterministic_torch: bool = False) -> int:
    """Seed every random number generator that is actually installed.

    NumPy and PyTorch are imported lazily: the template itself has no runtime
    dependencies, so this function must work whether or not they are present.

    Args:
        seed: Value applied to all generators.
        deterministic_torch: Force deterministic cuDNN kernels. Costs speed,
            so enable it only when bit-exact reruns matter more than throughput.

    Returns:
        The seed that was applied, so callers can log it alongside results.
    """
    random.seed(seed)
    # Hash randomization affects set/dict iteration order in subprocesses.
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return seed
