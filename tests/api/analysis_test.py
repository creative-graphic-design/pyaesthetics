import io
from typing import get_args

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pyaesthetics.analysis import AnalyzeMethod, ImageAnalysisOutput, analyze_image
from pyaesthetics.utils import PyaestheticsTestCase
from pyaesthetics.utils.typehint import PilImage


class TestAnalysisEndpoint(PyaestheticsTestCase):
    @pytest.fixture
    def image_filename(self) -> str:
        return "sample2.jpg"

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
        argnames="method",
        argvalues=get_args(AnalyzeMethod),
    )
    def test_analyze_image_endpoint(
        self,
        client: TestClient,
        method: AnalyzeMethod,
        image: PilImage,
        image_io: io.BytesIO,
        image_filename: str,
    ):
        res = client.post(
            "/analysis/image",
            files={"image_file": (image_filename, image_io)},
            data={"method": method},
        )
        res.raise_for_status()

        actual = ImageAnalysisOutput(**res.json())
        breakpoint()
        expected = analyze_image(image, method=method)
        assert actual == expected
