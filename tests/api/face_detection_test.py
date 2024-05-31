import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pyaesthetics.face_detection import GetFacesOutput, get_faces
from pyaesthetics.utils import PyaestheticsTestCase
from pyaesthetics.utils.typehint import PilImage


class TestFaceDetectionEndpoint(PyaestheticsTestCase):
    @pytest.fixture
    def image_filename(self) -> str:
        return "turing-2018-bengio-hinton-lecun.jpg"

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
        argnames="is_plot",
        argvalues=(True, False),
    )
    def test_opencv_endpoint(
        self,
        client: TestClient,
        is_plot: bool,
        image: PilImage,
        image_io: io.BytesIO,
        image_filename: str,
    ):
        res = client.post(
            "/face-detection/opencv",
            files={"image_file": (image_filename, image_io)},
            params={"is_plot": is_plot},
        )
        res.raise_for_status()

        actual = GetFacesOutput(**res.json())
        expected = get_faces(image, is_plot=is_plot)
        assert actual == expected
