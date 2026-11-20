from ..base import Accelerator


class ROCmAccelerator(Accelerator):
    name = "rocm"

    def is_available(self) -> bool:
        torch = self._torch()
        if torch is None:
            return False
        cuda = getattr(torch, "cuda", None)
        version = getattr(torch, "version", None)
        return bool(
            cuda
            and callable(getattr(cuda, "is_available", None))
            and getattr(version, "hip", None) is not None
            and cuda.is_available()
        )

    def device_count(self) -> int:
        if not self.is_available():
            return 0

        return self._torch().cuda.device_count()

    def get_device(self, index: int = 0):
        if not self.is_available():
            raise RuntimeError("ROCm accelerator is not available.")
        self._validate_index(index)
        return self._torch().device(f"cuda:{index}")

    def device_name(self, index: int = 0) -> str:
        if not self.is_available():
            raise RuntimeError("ROCm accelerator is not available.")
        self._validate_index(index)
        return self._torch().cuda.get_device_name(index)

    @staticmethod
    def _torch():
        try:
            import torch
            return torch
        except (ImportError, ModuleNotFoundError):
            return None

    def _validate_index(self, index: int) -> None:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("Device index must be an integer.")
        count = self.device_count()
        if index < 0 or index >= count:
            raise ValueError(
                f"Invalid ROCm device index {index}. "
                f"Available devices: 0-{count - 1}"
            )