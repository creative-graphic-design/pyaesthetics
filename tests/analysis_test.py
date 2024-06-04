from typing import get_args

import pytest
from PIL import Image

from pyaesthetics.analysis import AnalyzeMethod, ImageAnalysisOutput, analyze_image
from pyaesthetics.utils import PyaestheticsTestCase
from pyaesthetics.utils.typehint import PilImage


class TestAnalysis(PyaestheticsTestCase):
    @pytest.fixture
    def image_filename(self) -> str:
        return "sample2.jpg"

    @pytest.fixture
    def image(self, image_filename: str) -> PilImage:
        return Image.open(self.FIXTURES_ROOT / image_filename)

    @pytest.mark.parametrize(
        argnames="method",
        argvalues=get_args(AnalyzeMethod),
    )
    @pytest.mark.parametrize(
        argnames="is_plot",
        argvalues=[True, False],
    )
    def test_analyze_image(self, method: AnalyzeMethod, image: PilImage, is_plot: bool):
        output = analyze_image(image, method=method, is_plot=is_plot)
        assert isinstance(output, ImageAnalysisOutput)
