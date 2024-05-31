import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pyaesthetics.space_based_decomposition import (
    AreasOutput,
    TextImageRatioOutput,
    get_areas,
    get_text_image_ratio,
)
from pyaesthetics.utils import PyaestheticsTestCase
from pyaesthetics.utils.typehint import PilImage


class TestSpaceBasedDecompositionEndpoint(PyaestheticsTestCase):
    @pytest.fixture
    def image_filename(self) -> str:
        return "sample2.jpg"

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
        argnames="url",
        argvalues=(
            "/space-based-decomposition/areas",
            "/space-based-decomposition/areas/tesseract",
        ),
    )
    @pytest.mark.parametrize(
        argnames="is_plot",
        argvalues=(True, False),
    )
    @pytest.mark.parametrize(
        argnames="is_coordinates",
        argvalues=(True, False),
    )
    @pytest.mark.parametrize(
        argnames="is_areatype",
        argvalues=(True, False),
    )
    def test_get_areas_endpoint(
        self,
        client: TestClient,
        url: str,
        image: PilImage,
        image_io: io.BytesIO,
        image_filename: str,
        is_plot: bool,
        is_coordinates: bool,
        is_areatype: bool,
        min_area: int = 100,
        is_resize: bool = True,
        resized_w: int = 600,
        resized_h: int = 400,
    ):
        res = client.post(
            url=url,
            files={"image_file": (image_filename, image_io)},
            params={
                "min_area": min_area,
                "is_resize": is_resize,
                "resized_w": resized_w,
                "resized_h": resized_h,
                "is_plot": is_plot,
                "is_coordinates": is_coordinates,
                "is_areatype": is_areatype,
            },
        )
        res.raise_for_status()
        actual = AreasOutput(**res.json())

        expected = get_areas(
            image,
            min_area=min_area,
            is_resize=is_resize,
            resized_w=resized_w,
            resized_h=resized_h,
            is_plot=is_plot,
            is_coordinates=is_coordinates,
            is_areatype=is_areatype,
        )
        assert actual == expected

    @pytest.mark.parametrize(
        argnames="url",
        argvalues=(
            "/space-based-decomposition/text-image-ratio",
            "/space-based-decomposition/text-image-ratio/tesseract",
        ),
    )
    @pytest.mark.parametrize(
        argnames="is_plot",
        argvalues=(True, False),
    )
    @pytest.mark.parametrize(
        argnames="is_coordinates",
        argvalues=(True, False),
    )
    @pytest.mark.parametrize(
        argnames="is_areatype",
        argvalues=(True, False),
    )
    def test_get_text_image_ratio_endpoint(
        self,
        client: TestClient,
        url: str,
        image: PilImage,
        image_io: io.BytesIO,
        image_filename: str,
        is_plot: bool,
        is_coordinates: bool,
        is_areatype: bool,
        min_area: int = 100,
        is_resize: bool = True,
        resized_w: int = 600,
        resized_h: int = 400,
    ):
        res = client.post(
            url=url,
            files={"image_file": (image_filename, image_io)},
            params={
                "min_area": min_area,
                "is_resize": is_resize,
                "resized_w": resized_w,
                "resized_h": resized_h,
                "is_plot": is_plot,
                "is_coordinates": is_coordinates,
                "is_areatype": is_areatype,
            },
        )
        res.raise_for_status()
        actual = TextImageRatioOutput(**res.json())

        expected = get_text_image_ratio(
            areas_output=get_areas(
                image,
                min_area=min_area,
                is_resize=is_resize,
                resized_w=resized_w,
                resized_h=resized_h,
                is_plot=is_plot,
                is_coordinates=is_coordinates,
                is_areatype=is_areatype,
            )
        )
        assert actual == expected
