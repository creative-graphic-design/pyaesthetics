from fastapi import APIRouter

from pyaesthetics.api import routers
from pyaesthetics.utils.testing import PyaestheticsTestCase


class TestApiRouters(PyaestheticsTestCase):
    def test_num_routers_from_module(self, num_expected_routers: int = 9):
        num_routers = 0

        for cls in routers.__dict__.values():
            if isinstance(cls, APIRouter):
                num_routers += 1

        assert num_routers == num_expected_routers

    def test_num_routers_from_scripts(self):
        file_paths = list((self.MODULE_ROOT / "api" / "routers").glob("*.py"))

        num_files = len(file_paths)
        num_files -= 1  # exclude __init__.py

        expected_num_routers = len(routers.__all__)
        assert num_files == expected_num_routers
