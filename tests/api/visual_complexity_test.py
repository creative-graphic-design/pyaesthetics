import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pyaesthetics.utils import PyaestheticsTestCase
from pyaesthetics.utils.typehint import PilImage
from pyaesthetics.visual_complexity import VisualComplexityOutput, get_visual_complexity


class TestVisualComplexityEndpoint(PyaestheticsTestCase):
    @pytest.fixture
    def image_filename(self) -> str:
        return "sample.jpg"

    @pytest.fixture
    def image(self, image_filename: str) -> PilImage:
        sample_image_path = str(self.FIXTURES_ROOT / image_filename)
        return Image.open(sample_image_path)

    @pytest.fixture
    def image_io(self, image: PilImage) -> io.BytesIO:
        image_io = io.BytesIO()
        image.save(image_io, format="PNG")
        image_io.seek(0)
        return image_io

    @pytest.mark.parametrize(
        argnames="is_weight",
        argvalues=(True, False),
    )
    def test_visual_complexity_endpoint(
        self,
        client: TestClient,
        image: PilImage,
        image_io: io.BytesIO,
        image_filename: str,
        is_weight: bool,
        min_std: int = 15,
        min_size: int = 40,
    ):
        res = client.post(
            "/visual-complexity/",
            files={"image_file": (image_filename, image_io)},
            params={"is_weight": is_weight, "min_std": min_std, "min_size": min_size},
        )
        res.raise_for_status()

        actual = VisualComplexityOutput(**res.json())
        expected = get_visual_complexity(
            image, min_std=min_std, min_size=min_size, is_weight=is_weight
        )
        assert actual == expected
