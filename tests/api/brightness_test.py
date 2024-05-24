import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pyaesthetics.brightness import get_relative_luminance_bt601, get_relative_luminance_bt709
from pyaesthetics.utils import PyaestheticsTestCase
from pyaesthetics.utils.typehint import PilImage


class TestBrightnessEndpoint(PyaestheticsTestCase):
    @pytest.fixture
    def image_filename(self) -> str:
        return "sample.jpg"

    @pytest.fixture
    def image(self, image_filename: str) -> PilImage:
        return Image.open(self.FIXTURES_ROOT / image_filename)

    @pytest.fixture
    def image_io(self, image: PilImage) -> io.BytesIO:
        image_io = io.BytesIO()
        image.save(image_io, format="PNG")
        image_io.seek(0)
        return image_io

    def test_bt601_endpoint(
        self,
        client: TestClient,
        image: PilImage,
        image_io: io.BytesIO,
        image_filename: str,
    ):
        res = client.post(
            "/brightness/bt601",
            files={"image_file": (image_filename, image_io)},
        )
        res.raise_for_status()

        actual = res.json()
        expected = get_relative_luminance_bt601(image)
        assert actual == expected

    def test_bt709_endpoint(
        self,
        client: TestClient,
        image: PilImage,
        image_io: io.BytesIO,
        image_filename: str,
    ):
        res = client.post(
            "/brightness/bt709",
            files={"image_file": (image_filename, image_io)},
        )
        res.raise_for_status()

        actual = res.json()
        expected = get_relative_luminance_bt709(image)
        assert actual == expected
