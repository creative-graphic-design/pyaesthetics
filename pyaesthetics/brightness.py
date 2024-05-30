"""
Provides functions to evaluate the brightness of an image according to both BT.709 and BT.601 standards.

The module includes a converter for sRGB to RGB, and functions to calculate the relative luminance of an image
according to BT.709 and BT.601 standards. The brightness output is represented using a Pydantic model class.

@author: Giulio Gabrieli, Shunsuke Kitada

Classes
-------
BrightnessOutput : BaseModel
    A Pydantic model class that represents the brightness output.

Functions
---------
get_relative_luminance_bt709(img: PilImage) -> float
    Calculates the relative luminance of an image using the BT.709 standard.
get_relative_luminance_bt601(img: PilImage) -> float
    Calculates the relative luminance of an image using the BT.601 standard.
"""  # NOQA: E501

from typing import Optional

import numpy as np
from pydantic import BaseModel

from pyaesthetics.utils import s_rgb_to_rgb
from pyaesthetics.utils.typehint import PilImage


class BrightnessOutput(BaseModel):
    """
    A Pydantic model class that represents the brightness output.

    Attributes
    ----------
    bt709 : float
        The BT.709 brightness output value. BT.709 is a standard for high-definition
        digital TV broadcast and is used for HDTV systems. The value should be a float.

    bt601 : float, optional
        The BT.601 brightness output value, by default None. BT.601 is a standard for
        standard-definition digital color TV and is used for SDTV systems. The value should
        be a float if provided.
    """

    bt709: float
    bt601: Optional[float] = None


def get_relative_luminance_bt709(img: PilImage) -> float:
    """
    Calculate the relative luminance of an image using the BT.709 standard.

    This function evaluates the brightness of an image by means of Y, where Y
    is evaluated as:

    Y = 0.7152G + 0.0722B + 0.2126R
    B = mean(Y)

    The image must be in RGB mode.

    Parameters
    ----------
    img : PilImage
        The image to analyze, in RGB.

    Returns
    -------
    float
        The mean brightness of the image.

    Raises
    ------
    AssertionError
        If the image is not in RGB mode.

    Notes
    -----
    The BT.709 standard is used for high-definition digital TV broadcasts and
    is used for HDTV systems.

    Examples
    --------
    >>> img = Image.open('example.jpg')
    >>> get_relative_luminance_bt709(img)
    0.567
    """  # NOQA: E501
    assert img.mode == "RGB", "Image must be in RGB mode"

    img_arr = np.array(img)
    img_arr = s_rgb_to_rgb(img_arr)

    img_arr = img_arr.flatten()
    img_arr = img_arr.reshape(int(len(img_arr) / 3), 3)
    img_arr = np.transpose(img_arr)

    brigthness = (
        np.mean(img_arr[0]) * 0.2126 + np.mean(img_arr[1]) * 0.7152 + np.mean(img_arr[2]) * 0.0722
    )
    return brigthness.item()


def get_relative_luminance_bt601(img: PilImage) -> float:
    """
    Calculate the relative luminance of an image using the BT.601 standard.

    This function evaluates the brightness of an image by means of Y, where Y
    is evaluated as:

    Y = 0.587G + 0.114B + 0.299R
    B = mean(Y)

    The image must be in RGB mode.

    Parameters
    ----------
    img : PilImage
        The image to analyze, in RGB.

    Returns
    -------
    float
        The mean brightness of the image.

    Raises
    ------
    AssertionError
        If the image is not in RGB mode.

    Notes
    -----
    The BT.601 standard is used for standard-definition digital color TV and
    is used for SDTV systems.

    Examples
    --------
    >>> img = Image.open('example.jpg')
    >>> get_relative_luminance_bt601(img)
    0.467
    """
    assert img.mode == "RGB", "Image must be in RGB mode"

    img_arr = np.array(img)
    img_arr = s_rgb_to_rgb(img_arr)

    img_arr = img_arr.flatten()
    img_arr = img_arr.reshape(int(len(img_arr) / 3), 3)
    img_arr = np.transpose(img_arr)

    brightness = (
        np.mean(img_arr[0]) * 0.299 + np.mean(img_arr[1]) * 0.587 + np.mean(img_arr[2]) * 0.114
    )
    return brightness.item()
