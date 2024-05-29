import numpy as np
import pytest
from PIL import Image

from pyaesthetics.symmetry import get_symmetry
from pyaesthetics.utils import PyaestheticsTestCase
from pyaesthetics.utils.typehint import Base64EncodedImage, PilImage


class TestSymmetry(PyaestheticsTestCase):
    @pytest.fixture
    def image_filename(self) -> str:
        return "sample.jpg"

    @pytest.fixture
    def image(self, image_filename: str) -> PilImage:
        sample_image_path = str(self.FIXTURES_ROOT / image_filename)
        return Image.open(sample_image_path)

    @pytest.mark.parametrize(
        argnames="min_std, min_size, expected_result",
        argvalues=(
            (
                5,
                20,
                74.43491816056118,  # 60.747663551401864,
            ),
        ),
    )
    def test_symmetry(self, image: PilImage, min_std: int, min_size: int, expected_result: float):
        output = get_symmetry(image, min_std=min_std, min_size=min_size)
        assert not isinstance(output.degree, np.floating)

        actual_result = output.degree
        assert actual_result == expected_result
        assert output.images is None

    @pytest.mark.parametrize(
        argnames="min_std, min_size",
        argvalues=((5, 20),),
    )
    def test_plot(self, image: PilImage, min_std: int, min_size: int):
        output = get_symmetry(image, min_std=min_std, min_size=min_size, is_plot=True)
        assert output.images is not None

        assert isinstance(output.images.left, Base64EncodedImage)
        assert isinstance(output.images.right, Base64EncodedImage)
