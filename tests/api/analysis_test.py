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
        is_resize: bool = True,
        new_size_w: int = 600,
        new_size_h: int = 400,
        min_std: int = 10,
        min_size: int = 20,
    ):
        res = client.post(
            "/analysis/image",
            files={"image_file": (image_filename, image_io)},
            params={
                "method": method,
                "is_resize": is_resize,
                "new_size_w": new_size_w,
                "new_size_h": new_size_h,
                "min_std": min_std,
                "min_size": min_size,
            },
        )
        res.raise_for_status()

        actual = ImageAnalysisOutput(**res.json())
        expected = analyze_image(
            image,
            method=method,
            is_resize=is_resize,
            resized_w=new_size_w,
            resized_h=new_size_h,
            min_std=min_std,
            min_size=min_size,
        )
        assert actual == expected
