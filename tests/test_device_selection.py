import unittest
from unittest.mock import patch

import torch

from src.lsm.trainer import get_device


class DeviceSelectionTest(unittest.TestCase):
    def test_auto_prefers_cuda(self):
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
            torch.backends.mps, "is_available", return_value=True
        ):
            self.assertEqual(get_device("auto").type, "cuda")

    def test_auto_uses_mps_when_cuda_unavailable(self):
        with patch.object(torch.cuda, "is_available", return_value=False), patch.object(
            torch.backends.mps, "is_available", return_value=True
        ):
            self.assertEqual(get_device("auto").type, "mps")

    def test_cuda_request_fails_when_unavailable(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "device='cuda'"):
                get_device("cuda")

    def test_cpu_request(self):
        self.assertEqual(get_device("cpu").type, "cpu")


if __name__ == "__main__":
    unittest.main()
