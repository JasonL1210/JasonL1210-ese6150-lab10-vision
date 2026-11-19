from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class EntryPointTests(unittest.TestCase):
    def test_required_entry_points_import_without_optional_ml_packages(self) -> None:
        for module_name in ("camera", "convert_trt", "detection", "integrated"):
            with self.subTest(module_name=module_name):
                module = importlib.import_module(module_name)
                self.assertTrue(callable(module.main))

    def test_simulation_config_check_is_available_without_hardware(self) -> None:
        camera = importlib.import_module("camera")
        integrated = importlib.import_module("integrated")
        self.assertEqual(camera.main(["--mode", "simulation", "--check-config"]), 0)
        self.assertEqual(integrated.main(["--mode", "simulation", "--check-config"]), 0)


if __name__ == "__main__":
    unittest.main()

