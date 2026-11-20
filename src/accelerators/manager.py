from .detector import detect_accelerator
from .cuda.backend import CUDAAccelerator
from .rocm.backend import ROCmAccelerator
from .cpu.backend import CPUAccelerator


class AcceleratorManager:

    def __init__(self, backend_name: str | None = None):
        self.backend_name = (
            detect_accelerator() if backend_name is None else backend_name.lower()
        )
        self.backend = self._create_backend()

    def _create_backend(self):
        if self.backend_name == "cuda":
            return CUDAAccelerator()

        if self.backend_name == "rocm":
            return ROCmAccelerator()

        if self.backend_name == "cpu":
            return CPUAccelerator()

        raise ValueError(
            f"Unknown accelerator backend: {self.backend_name}. "
            "Expected one of: cpu, cuda, rocm"
        )

    def get_backend(self):
        return self.backend

    def get_device(self, index: int = 0):
        return self.backend.get_device(index)

    def device_count(self) -> int:
        return self.backend.device_count()

    def print_summary(self):
        print(f"Backend: {self.backend.name}")
        print(f"Available: {self.backend.is_available()}")
        print(f"Devices: {self.backend.device_count()}")

        for index in range(self.backend.device_count()):
            print(
                f"  [{index}] {self.backend.device_name(index)}"
            )