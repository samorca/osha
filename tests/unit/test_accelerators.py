import sys
import types
import unittest
from unittest.mock import patch

from src.accelerators import AcceleratorManager, detect_accelerator
from src.accelerators.cpu.backend import CPUAccelerator
from src.accelerators.cuda.backend import CUDAAccelerator
from src.accelerators.rocm.backend import ROCmAccelerator
from src.accelerators.cpu import devices as cpu_devices
from src.accelerators.cpu import metrics as cpu_metrics
from src.accelerators.cuda import devices as cuda_devices
from src.accelerators.cuda import metrics as cuda_metrics
from src.accelerators.rocm import devices as rocm_devices
from src.accelerators.rocm import metrics as rocm_metrics


def fake_torch(cuda_available=True, hip=None, cuda="12.1", device_count=2):
    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(cuda=cuda, hip=hip)
    properties = types.SimpleNamespace(
        name="Test GPU",
        total_memory=8 * 1024**3,
        major=8,
        minor=9,
    )
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        device_count=lambda: device_count,
        get_device_name=lambda index: f"device-{index}",
        get_device_properties=lambda device: properties,
        mem_get_info=lambda device: (
            2 * 1024**3,
            8 * 1024**3,
        ),
        memory_allocated=lambda device: 1 * 1024**3,
        memory_reserved=lambda device: 2 * 1024**3,
    )
    torch.device = lambda name: name
    torch.get_num_threads = lambda: 4
    torch.get_num_interop_threads = lambda: 2
    return torch


class AcceleratorTests(unittest.TestCase):
    def test_detects_cpu_when_pytorch_is_missing(self):
        with patch.dict(sys.modules, {"torch": None}):
            self.assertEqual(detect_accelerator(), "cpu")

    def test_detects_cuda(self):
        torch = fake_torch()
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertEqual(detect_accelerator(), "cuda")

    def test_detects_rocm(self):
        torch = fake_torch(hip="6.1", cuda=None)
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertEqual(detect_accelerator(), "rocm")

    def test_cpu_backend_validation(self):
        backend = CPUAccelerator()
        self.assertTrue(backend.is_available())
        self.assertEqual(backend.device_count(), 1)
        self.assertEqual(backend.device_name(0), "CPU")
        with self.assertRaises(ValueError):
            backend.get_device(-1)
        with self.assertRaises(TypeError):
            backend.get_device(True)

    def test_cuda_backend_validation(self):
        torch = fake_torch()
        with patch.dict(sys.modules, {"torch": torch}):
            backend = CUDAAccelerator()
            self.assertTrue(backend.is_available())
            self.assertEqual(backend.device_count(), 2)
            self.assertEqual(backend.get_device(1), "cuda:1")
            with self.assertRaises(ValueError):
                backend.get_device(-1)
            with self.assertRaises(ValueError):
                backend.get_device(2)

    def test_rocm_backend_validation(self):
        torch = fake_torch(hip="6.1", cuda=None)
        with patch.dict(sys.modules, {"torch": torch}):
            backend = ROCmAccelerator()
            self.assertTrue(backend.is_available())
            self.assertEqual(backend.get_device(0), "cuda:0")

    def test_manager_override_and_unknown_backend(self):
        manager = AcceleratorManager("cpu")
        self.assertIsInstance(manager.get_backend(), CPUAccelerator)
        with self.assertRaises(ValueError):
            AcceleratorManager("unknown")

    def test_cpu_devices_and_metrics(self):
        torch = fake_torch(cuda_available=False)
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertTrue(cpu_devices.is_cpu_available())
            self.assertGreaterEqual(cpu_devices.get_device_count(), 1)
            self.assertEqual(cpu_devices.get_device(), "cpu")
            self.assertEqual(len(cpu_devices.list_devices()), 1)
            info = cpu_devices.get_device_info()
            self.assertEqual(info["backend"], "cpu")
            self.assertEqual(info["pytorch_threads"], 4)

        fake_psutil = types.SimpleNamespace(
            cpu_percent=lambda interval=0.1, percpu=False: (
                [10.0, 20.0] if percpu else 15.0
            ),
            cpu_count=lambda logical=True: 8,
            cpu_freq=lambda: types.SimpleNamespace(current=2400.0),
            getloadavg=lambda: (1.0, 1.5, 2.0),
            virtual_memory=lambda: types.SimpleNamespace(
                total=16 * 1024**3,
                available=8 * 1024**3,
                used=8 * 1024**3,
                percent=50.0,
            ),
            swap_memory=lambda: types.SimpleNamespace(
                total=4 * 1024**3,
                used=1 * 1024**3,
                free=3 * 1024**3,
                percent=25.0,
            ),
        )
        with patch.object(cpu_metrics, "psutil", fake_psutil):
            cpu = cpu_metrics.get_cpu_metrics()
            memory = cpu_metrics.get_memory_metrics()
            swap = cpu_metrics.get_swap_metrics()
            all_metrics = cpu_metrics.get_all_metrics()
        self.assertEqual(cpu["utilization_percent"], 15.0)
        self.assertEqual(memory["memory_utilization_percent"], 50.0)
        self.assertEqual(swap["swap_utilization_percent"], 25.0)
        self.assertIn("cpu", all_metrics)

    def test_missing_cpu_optional_dependencies(self):
        with patch.dict(sys.modules, {"torch": None}):
            self.assertIsInstance(cpu_devices.get_device(), str)
            self.assertEqual(cpu_devices.get_device_info()["pytorch_threads"], None)

        with patch.object(cpu_metrics, "psutil", None):
            self.assertEqual(
                cpu_metrics.get_cpu_metrics()["logical_cores"],
                cpu_devices.get_device_count(),
            )
            self.assertEqual(
                cpu_metrics.get_memory_metrics()["total_memory_bytes"],
                cpu_metrics.get_memory_metrics()["available_memory_bytes"],
            )
            self.assertEqual(cpu_metrics.get_swap_metrics()["total_swap_bytes"], 0)

    def test_cuda_devices_and_metrics(self):
        torch = fake_torch()
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertTrue(cuda_devices.is_cuda_available())
            self.assertEqual(cuda_devices.get_device_count(), 2)
            self.assertEqual(cuda_devices.get_device(1), "cuda:1")
            info = cuda_devices.get_device_info(0)
            self.assertEqual(info["backend"], "cuda")
            self.assertEqual(info["compute_capability"], (8, 9))
            self.assertEqual(len(cuda_devices.list_devices()), 2)
            with self.assertRaises(TypeError):
                cuda_devices.get_device(True)
            with self.assertRaises(ValueError):
                cuda_devices.get_device(2)

            metrics = cuda_metrics.get_memory_metrics(0)
            all_metrics = cuda_metrics.get_all_memory_metrics()
        self.assertEqual(metrics["backend"], "cuda")
        self.assertEqual(metrics["used_memory_bytes"], 6 * 1024**3)
        self.assertEqual(len(all_metrics), 2)

    def test_rocm_devices_and_metrics(self):
        torch = fake_torch(hip="6.1", cuda=None)
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertTrue(rocm_devices.is_rocm_available())
            self.assertEqual(rocm_devices.get_device_count(), 2)
            self.assertEqual(rocm_devices.get_device(1), "cuda:1")
            info = rocm_devices.get_device_info(0)
            self.assertEqual(info["backend"], "rocm")
            self.assertEqual(info["rocm_version"], "6.1")
            self.assertEqual(len(rocm_devices.list_devices()), 2)

            metrics = rocm_metrics.get_memory_metrics(0)
            all_metrics = rocm_metrics.get_all_memory_metrics()
        self.assertEqual(metrics["backend"], "rocm")
        self.assertEqual(len(all_metrics), 2)

    def test_unavailable_gpu_helpers(self):
        unavailable = fake_torch(cuda_available=False)
        with patch.dict(sys.modules, {"torch": unavailable}):
            self.assertFalse(cuda_devices.is_cuda_available())
            self.assertFalse(rocm_devices.is_rocm_available())
            self.assertEqual(cuda_devices.get_device_count(), 0)
            self.assertEqual(rocm_devices.get_device_count(), 0)
            with self.assertRaises(RuntimeError):
                cuda_devices.get_device()
            with self.assertRaises(RuntimeError):
                rocm_devices.get_device()

    def test_detector_handles_missing_cuda_api(self):
        torch = types.ModuleType("torch")
        torch.version = types.SimpleNamespace(cuda="12.1", hip=None)
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertEqual(detect_accelerator(), "cpu")


if __name__ == "__main__":
    unittest.main()
