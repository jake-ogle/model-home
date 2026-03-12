"""Device detection utility for running models on CUDA, MPS, or CPU."""

import torch


def get_device(device=None):
    """
    Detect and return the appropriate device for PyTorch computations.

    Priority:
    1. If device is explicitly specified, use that
    2. CUDA (NVIDIA GPUs)
    3. MPS (Apple Silicon)
    4. CPU (fallback)

    Args:
        device (str or torch.device, optional): Explicit device specification.
            Can be "cuda", "mps", "cpu", or a torch.device object.
            If None, auto-detects the best available device.

    Returns:
        torch.device: The device to use for computation

    Examples:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device("cuda")  # Force CUDA
        >>> device = get_device("cpu")  # Force CPU
    """
    if device is not None:
        if isinstance(device, str):
            return torch.device(device)
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
