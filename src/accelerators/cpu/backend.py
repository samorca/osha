from typing import Any

from ..base import Accelerator


class CPUAccelerator(Accelerator):
    name = "cpu"

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def get_device(self, index: int = 0) -> Any:
        self._validate_index(index)
        try:
            import torch
            return torch.device("cpu")
        except (ImportError, ModuleNotFoundError):
            return "cpu"

    def device_name(self, index: int = 0) -> str:
        self._validate_index(index)
        return "CPU"

    def _validate_index(self, index: int) -> None:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("Device index must be an integer.")
        if index != 0:
            raise ValueError(
                f"CPU device index {index} is invalid. Available devices: 0"
            )