import pytest
from fastapi import APIRouter

from pyaesthetics.api import routers
from pyaesthetics.api.routers import gather_routers
from pyaesthetics.utils.testing import PyaestheticsTestCase


class TestApiRouters(PyaestheticsTestCase):
    @pytest.fixture
    def num_expected_routers(self) -> int:
        return 9

    def test_num_routers_from_module(self, num_expected_routers: int):
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

    def test_gather_routers(self, num_expected_routers: int):
        routers = gather_routers()
        assert len(routers) == num_expected_routers
