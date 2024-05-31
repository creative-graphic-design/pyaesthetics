"""
Provides a function for evaluating the saturation of an image.

The module includes a function to calculate the average saturation of an image in the HSV color space. The image
must be in RGB mode.

@author: Giulio Gabrieli, Shunsuke Kitada

Functions
---------
get_saturation(img: PilImage) -> float
    Calculate the average saturation of an image.
"""  # NOQA: E501

import cv2
import numpy as np

from pyaesthetics.utils.typehint import PilImage


def get_saturation(img: PilImage) -> float:
    """
    Calculate the average saturation of an image.

    This function converts the image to the HSV color space, and then calculates the average of the S values
    (saturation) in the HSV color space.

    The image must be in RGB mode.

    Parameters
    ----------
    img : PilImage
        The image to analyze, in RGB.

    Returns
    -------
    float
        The average saturation of the image.

    Raises
    ------
    AssertionError
        If the image is not in RGB mode.

    Examples
    --------
    >>> img = Image.open('example.jpg')
    >>> get_saturation(img)
    0.567
    """  # NOQA: E501
    assert img.mode == "RGB", "Image must be in RGB mode"
    img_arr = np.array(img)
    img_hsv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
    img_hsv = np.divide(img_hsv, 255)
    saturation = img_hsv[:, :, 1].mean()

    return saturation.item()
