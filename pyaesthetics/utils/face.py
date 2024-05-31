"""
Provides abstract and concrete classes for face detection.

The `FaceDetector` class is an abstract base class that defines the interface for a face detector.
Subclasses should implement the `__call__` method to detect faces in an image and the `plot_bboxes` method
to visualize the detected bounding boxes on the image.

The `Cv2CancadeClassifier` class is a concrete implementation of the `FaceDetector` interface using
OpenCV's Cascade Classifier for face detection. The classifier uses a Haar cascade trained on frontal faces.

Classes
-------
FaceDetector
    Abstract base class for face detection.

Cv2CancadeClassifier
    Face detection using OpenCV's Cascade Classifier.

Example
-------
To use this module, import it and instantiate an object of the `Cv2CancadeClassifier` class:

    import cv2_face_detection_module
    from PIL import Image

    face_detector = cv2_face_detection_module.Cv2CancadeClassifier()
    image = Image.open('face_image.jpg')
    bboxes = face_detector(image)
    images_with_bboxes = face_detector.plot_bboxes(image, bboxes)
"""  # NOQA: E501

import abc
from typing import List, Tuple

import cv2
import numpy as np
from PIL import ImageDraw

from pyaesthetics.utils.typehint import PilImage


class FaceDetector(object, metaclass=abc.ABCMeta):
    """
    Abstract base class for face detection.

    This class defines the interface for a face detector. Subclasses should implement
    the `__call__` method to detect faces in an image and the `plot_bboxes` method to
    visualize the detected bounding boxes on the image.

    Methods
    -------
    plot_bboxes(image: PilImage, bboxes: List[Tuple[int, int, int, int]]) -> List[PilImage]:
        Draw bounding boxes on the image.

    __call__(image: PilImage) -> List[Tuple[int, int, int, int]]:
        Detect faces in the image.
    """

    @abc.abstractmethod
    def plot_bboxes(
        self, image: PilImage, bboxes: List[Tuple[int, int, int, int]]
    ) -> List[PilImage]:
        """
        Draw bounding boxes on the image.

        Parameters
        ----------
        image : PilImage
            The input image.
        bboxes : List[Tuple[int, int, int, int]]
            A list of bounding boxes, where each box is a tuple of (x, y, width, height).

        Returns
        -------
        List[PilImage]
            A list of images with bounding boxes drawn on them.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, image: PilImage) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the image.

        Parameters
        ----------
        image : PilImage
            The input image.

        Returns
        -------
        List[Tuple[int, int, int, int]]
            A list of bounding boxes, where each box is a tuple of (x, y, width, height).
        """
        raise NotImplementedError


class Cv2CancadeClassifier(FaceDetector):
    """
    Face detection using OpenCV's Cascade Classifier.

    This class implements the FaceDetector interface using OpenCV's Cascade Classifier
    for face detection. The classifier uses a Haar cascade trained on frontal faces.

    Attributes
    ----------
    cascade : cv2.CascadeClassifier
        The Haar cascade classifier for face detection.

    Methods
    -------
    plot_bboxes(image: PilImage, bboxes: List[Tuple[int, int, int, int]]) -> List[PilImage]:
        Draw bounding boxes on the image.

    __call__(image: PilImage) -> List[Tuple[int, int, int, int]]:
        Detect faces in the image.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cascade = cv2.CascadeClassifier(
            filename=cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
        )

    def plot_bboxes(
        self, image: PilImage, bboxes: List[Tuple[int, int, int, int]]
    ) -> List[PilImage]:
        """
        Draw bounding boxes on the image.

        Parameters
        ----------
        image : PilImage
            The input image.
        bboxes : List[Tuple[int, int, int, int]]
            A list of bounding boxes, where each box is a tuple of (x, y, width, height).

        Returns
        -------
        List[PilImage]
            A list of images with bounding boxes drawn on them.
        """
        images = []
        for x, y, w, h in bboxes:
            img_copy = image.copy()
            draw = ImageDraw.Draw(img_copy)
            draw.rectangle((x, y, x + w, y + h), outline="red")
            images.append(img_copy)
        return images

    def __call__(self, image: PilImage) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the image.

        Parameters
        ----------
        image : PilImage
            The input image in RGB mode.

        Returns
        -------
        List[Tuple[int, int, int, int]]
            A list of bounding boxes, where each box is a tuple of (x, y, width, height).

        Raises
        ------
        AssertionError
            If the image is not in RGB mode.
        """
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
