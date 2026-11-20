def detect_accelerator() -> str:
    """Detect the best available accelerator supported by PyTorch."""
    try:
        import torch
    except (ImportError, ModuleNotFoundError):
        return "cpu"

    cuda = getattr(torch, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        return "cpu"

    if not cuda.is_available():
        return "cpu"

    version = getattr(torch, "version", None)
    if getattr(version, "hip", None):
        return "rocm"
    if getattr(version, "cuda", None):
        return "cuda"
    return "cpu"