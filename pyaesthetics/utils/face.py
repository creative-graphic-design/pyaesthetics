import abc
from typing import List, Tuple

import cv2
import numpy as np
from PIL import ImageDraw
from PIL.Image import Image as PilImage


class FaceDetector(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def plot_bboxes(
        self, image: PilImage, bboxes: List[Tuple[int, int, int, int]]
    ) -> List[PilImage]:
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, image: PilImage) -> List[Tuple[int, int, int, int]]:
        raise NotImplementedError


class Cv2CancadeClassifier(FaceDetector):
    def __init__(self) -> None:
        super().__init__()
        self.cascade = cv2.CascadeClassifier(
            filename=cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
        )

    def plot_bboxes(
        self, image: PilImage, bboxes: List[Tuple[int, int, int, int]]
    ) -> List[PilImage]:
        images = []
        for x, y, w, h in bboxes:
            img_copy = image.copy()
            draw = ImageDraw.Draw(img_copy)
            draw.rectangle((x, y, x + w, y + h), outline="red")
            images.append(img_copy)
        return images

    def __call__(self, image: PilImage) -> List[Tuple[int, int, int, int]]:
        assert image.mode == "RGB", "Image must be in RGB mode"
        image_arr = np.array(image)
        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)

        faces_bboxes: np.ndarray = self.cascade.detectMultiScale(  # type: ignore
            image_arr,
            scaleFactor=1.3,
            minNeighbors=5,
        )
        bboxes = (
            faces_bboxes.tolist()
            if isinstance(faces_bboxes, np.ndarray)
            else list(faces_bboxes)  # empty result
        )
        return bboxes
