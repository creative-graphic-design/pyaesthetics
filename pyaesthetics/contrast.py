"""
Provides functions for evaluating the contrast of an image in both the RMS and Michelson methods.

The module includes functions to calculate the root mean square (RMS) contrast and the Michelson contrast of an image.
It also provides a Pydantic model class to represent the contrast output.

@author: Giulio Gabrieli, Shunsuke Kitada

Classes
-------
ContrastOutput : BaseModel
    A Pydantic model class that represents the contrast output.

Functions
---------
contrast_rms(img: PilImage) -> float
    Calculate the root mean square (RMS) contrast of an image.
contrast_michelson(img: PilImage) -> float
    Calculate the Michelson contrast of an image.
"""  # NOQA: E501

from typing import Optional

import cv2
import numpy as np
from pydantic import BaseModel

from pyaesthetics.utils.typehint import PilImage


class ContrastOutput(BaseModel):
    """
    A Pydantic model class that represents the contrast output.

    Attributes
    ----------
    rms : float
        The root mean square (RMS) contrast of the image. It should be a float.

    michelson : float, optional
        The Michelson contrast of the image, by default None. It should be a float if provided.
    """

    rms: float
    michelson: Optional[float] = None


def contrast_rms(img: PilImage) -> float:
    """
    Calculate the root mean square (RMS) contrast of an image.

    This function converts the image to grayscale, and then calculates the standard deviation of the pixel values,
    which is the RMS contrast.

    The image must be in RGB mode.

    Parameters
    ----------
    img : PilImage
        The image to analyze, in RGB.

    Returns
    -------
    float
        The RMS contrast of the image.

    Raises
    ------
    AssertionError
        If the image is not in RGB mode.

    Examples
    --------
    >>> img = Image.open('example.jpg')
    >>> contrast_rms(img)
    0.567
    """  # NOQA: E501
    assert img.mode == "RGB", "Image must be in RGB mode"

    img_arr = np.array(img)
    img_grey = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    img_grey = img_grey.astype(float) / 255.0  # Convert img_grey to float type
    contrast = img_grey.std()

    # should be the same as:
    # img_s = img_grey - img_grey.mean()
    # img_s = img_s**2
    # contrast2 = np.sqrt(img_s.sum() / (img_s.shape[0] * img_s.shape[1]))
    # print(contrast, contrast2)

    return contrast.item()


def contrast_michelson(img: PilImage):
    """
    Calculate the Michelson contrast of an image.

    This function converts the image to the YUV color space, and then calculates the Michelson contrast based on
    the Y channel (luma information). The Michelson contrast is defined as (ymax - ymin) / (ymax + ymin), where
    ymax and ymin are the maximum and minimum of Y, respectively.

    The image must be in RGB mode.

    Parameters
    ----------
    img : PilImage
        The image to analyze, in RGB.

    Returns
    -------
    float
        The Michelson contrast of the image.

    Raises
    ------
    AssertionError
        If the image is not in RGB mode.

    Examples
    --------
    >>> img = Image.open('example.jpg')
    >>> contrast_michelson(img)
    0.567
    """  # NOQA: E501
    assert img.mode == "RGB", "Image must be in RGB mode"

    img_arr = np.array(img)
    y = cv2.cvtColor(img_arr, cv2.COLOR_RGB2YUV)[:, :, 0]

    # compute min and max of Y
    ymin = float(np.min(y))
    ymax = float(np.max(y))
    # compute contrast
    contrast = (ymax - ymin) / (ymax + ymin)

    return contrast
