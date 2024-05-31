import io
from typing import get_args

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pyaesthetics.color_detection import ColorDetectionOutput, NColorType, get_colors_w3c
from pyaesthetics.utils import PyaestheticsTestCase
from pyaesthetics.utils.typehint import PilImage


class TestColorDetectionEndpoint(PyaestheticsTestCase):
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

    @pytest.mark.parametrize(
        argnames="n_colors",
        argvalues=get_args(NColorType),
    )
    @pytest.mark.parametrize(
        argnames="is_plot",
        argvalues=(True, False),
    )
    def test_get_colors_w3c(
        self,
        client: TestClient,
        n_colors: NColorType,
        is_plot: bool,
        image: PilImage,
        image_io: io.BytesIO,
        image_filename: str,
    ):
        res = client.post(
            "/color-detection",
            files={"image_file": (image_filename, image_io)},
            params={"n_colors": n_colors, "is_plot": is_plot},
        )
        res.raise_for_status()

        actual = ColorDetectionOutput(**res.json())
        expected = get_colors_w3c(image)
        assert actual == expected
