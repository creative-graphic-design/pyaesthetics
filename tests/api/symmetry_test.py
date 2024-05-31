import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pyaesthetics.symmetry import SymmetryOutput, get_symmetry
from pyaesthetics.utils import PyaestheticsTestCase
from pyaesthetics.utils.typehint import PilImage


class TestSymmetryEndpoint(PyaestheticsTestCase):
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
        argnames="is_plot",
        argvalues=(True, False),
    )
    def test_get_symmetry_endpoint(
        self,
        client: TestClient,
        image: PilImage,
        image_io: io.BytesIO,
        image_filename: str,
        is_plot: bool,
        min_std: int = 5,
        min_size: int = 20,
    ):
        res = client.post(
            url="/symmetry",
            files={"image_file": (image_filename, image_io)},
            params={"min_std": min_std, "min_size": min_size, "is_plot": is_plot},
        )
        res.raise_for_status()
        actual = SymmetryOutput(**res.json())

        expected = get_symmetry(
            image,
            min_std=min_std,
            min_size=min_size,
            is_plot=is_plot,
        )
        assert actual == expected
