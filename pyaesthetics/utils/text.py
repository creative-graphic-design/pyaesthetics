"""
Provides classes and functions for text detection in images.

The `TextDetector` class is an abstract base class that defines the interface for a text detector.
Subclasses should implement the `__call__` method to detect text in an image.

The `TesseractTextDetector` class implements the TextDetector interface using Tesseract for text detection.

The `detect_text` function takes an image and a text detector, applies the detector to the image,
and returns the length of the detected text. If no text detector is provided, it uses
the TesseractTextDetector by default.

Classes
-------
TextDetector
    Abstract base class for text detection.

TesseractTextDetector
    Text detection using Tesseract.

Functions
---------
detect_text(image: PilImage, text_detector: Optional[TextDetector] = None) -> int
    Detect the amount of text in an image.

Example
-------
To use this module, import it along with Pillow and pass your image to the `detect_text` function:

    import text_detection_module
    from PIL import Image

    img = Image.new('RGB', (60, 30), color = 'red')
    text_length = text_detection_module.detect_text(img)
"""  # NOQA: E501

import abc
from typing import Optional

from pyaesthetics.utils.typehint import PilImage


class TextDetector(object, metaclass=abc.ABCMeta):
    """
    Abstract base class for text detection.

    This class defines the interface for a text detector. Subclasses should implement
    the `__call__` method to detect text in an image.

    Methods
    -------
    __call__(image: PilImage, *args, **kwargs) -> str:
        Detect text in the image.
    """

    @abc.abstractmethod
    def __call__(self, image: PilImage, *args, **kwargs) -> str:
        """
        Detect text in the image.

        This method should be implemented by subclasses to detect text in the input image.

        Parameters
        ----------
        image : PilImage
            The input image as a Pillow Image object.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        str
            The detected text in the image.
        """
        raise NotImplementedError


class TesseractTextDetector(TextDetector):
    """
    Text detection using Tesseract.

    This class implements the TextDetector interface using Tesseract for text detection.

    Methods
    -------
    __call__(image: PilImage, *args, **kwargs) -> str:
        Detect text in the image using Tesseract.
    """

    def __call__(self, image: PilImage, *args, **kwargs) -> str:
        """
        Detect text in the image using Tesseract.

        This method takes an image, applies Tesseract to it, and returns the detected text.

        Parameters
        ----------
        image : PilImage
            The input image as a Pillow Image object.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        str
            The detected text in the image.
        """
        import pytesseract

        text = pytesseract.image_to_string(image)
        return text


def detect_text(image: PilImage, text_detector: Optional[TextDetector] = None) -> int:
    """
    Detect the amount of text in an image.

    This function takes an image and a text detector, applies the detector to the image,
    and returns the length of the detected text. If no text detector is provided, it uses
    the TesseractTextDetector by default.

    Parameters
    ----------
    image : PilImage
        The input image as a Pillow Image object.
    text_detector : Optional[TextDetector], optional
        The text detector to use, default is TesseractTextDetector.

    Returns
    -------
    int
        The length of the detected text in the image.

    Examples
    --------
    >>> from PIL import Image
    >>> img = Image.open('text_image.jpg')
    >>> text_length = detect_text(img)
    >>> type(text_length)
    <class 'int'>
    """
    text_detector = text_detector or TesseractTextDetector()
    return len(text_detector(image))
