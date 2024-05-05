import pytest
from PIL import Image
from PIL.Image import Image as PilImage

from pyaesthetics.brightness import (
    relative_luminance_bt601,
    relative_luminance_bt709,
)
from pyaesthetics.utils import PyaestheticsTestCase


class TestBrightness(PyaestheticsTestCase):
    @pytest.fixture
    def image(self) -> PilImage:
        img_path = self.FIXTURES_ROOT / "sample.jpg"
        img = Image.open(img_path)
        return img

    def test_relative_luminance_bt601(self, image):
        brigtness = relative_luminance_bt601(image)
        assert pytest.approx(brigtness, 0.0001) == 0.6024

    def test_relative_luminance_bt709(self, image):
        brigtness = relative_luminance_bt709(image)
        assert pytest.approx(brigtness, 0.0001) == 0.5918