"""
Provides functions for evaluating the colorfulness of an image in both the HSV and RGB color spaces.

The module includes functions to calculate the colorfulness of an image using different formulas, including the
formula described in Yendrikhovskij et al., 1998 for the HSV color space, and Metric 3 described in Hasler &
Suesstrunk, 2003 for the RGB color space. It also provides a Pydantic model class to represent the colorfulness output.

@author: Giulio Gabrieli, Shunsuke Kitada

Classes
-------
ColorfulnessOutput : BaseModel
    A Pydantic model class that represents the colorfulness output.

Functions
---------
get_colorfulness_hsv(img: PilImage) -> float
    Calculate the colorfulness of an image in the HSV color space.
get_colorfulness_rgb(img: PilImage) -> float
    Calculate the colorfulness of an image in the RGB color space.
"""  # NOQA: E501

from typing import Optional

import cv2
import numpy as np
from pydantic import BaseModel

from pyaesthetics.utils.typehint import PilImage


class ColorfulnessOutput(BaseModel):
    """
    A Pydantic model class that represents the colorfulness output.

    Attributes
    ----------
    rgb : float
        The colorfulness value in the RGB color space. It should be a float.

    hsv : float, optional
        The colorfulness value in the HSV color space, by default None. It should be a float if provided.
    """  # NOQA: E501

    rgb: float
    hsv: Optional[float] = None


def get_colorfulness_hsv(img: PilImage) -> float:
    """
    Calculate the colorfulness of an image in the HSV color space.

    This function evaluates the colorfulness of a picture using the formula described in Yendrikhovskij et al., 1998.
    The input image is first converted to the HSV color space, then the S values are selected.
    The colorfulness index is evaluated with a sum of the mean S and its standard deviation, as in:

    Ci = mean(Si)+ std(Si)

    The image must be in RGB mode.

    Parameters
    ----------
    img : PilImage
        The image to analyze, in RGB.

    Returns
    -------
    float
        The colorfulness index of the image in the HSV color space.

    Raises
    ------
    AssertionError
        If the image is not in RGB mode.

    Examples
    --------
    >>> img = Image.open('example.jpg')
    >>> get_colorfulness_hsv(img)
    0.567
    """  # NOQA: E501
    assert img.mode == "RGB", "Image must be in RGB mode"

    img_arr = np.array(img)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)

    saturations = []  # initialize a list
    for row in img_arr:  # for each row
        for pixel in row:  # for each pixel
            saturations.append(pixel[1])  # take only the Saturation value
    colorfulness = np.mean(saturations) + np.std(saturations)  # evaluate the colorfulness
    return colorfulness.item()  # return the colorfulness index


def get_colorfulness_rgb(img: PilImage) -> float:
    """
    Calculate the colorfulness of an image in the RGB color space.

    This function evaluates the colorfulness of a picture using Metric 3 described in Hasler & Suesstrunk, 2003.
    The colorfulness index is evaluated with the following formula:

    Ci =std(rgyb) + 0.3 mean(rgyb)   [Equation Y]
    std(rgyb) = sqrt(std(rg)^2+std(yb)^2)
    mean(rgyb) = sqrt(mean(rg)^2+mean(yb)^2)
    rg = R - G
    yb = 0.5(R+G) - B

    The image must be in RGB mode.

    Parameters
    ----------
    img : PilImage
        The image to analyze, in RGB.

    Returns
    -------
    float
        The colorfulness index of the image in the RGB color space.

    Raises
    ------
    AssertionError
        If the image is not in RGB mode.

    Examples
    --------
    >>> img = Image.open('example.jpg')
    >>> get_colorfulness_rgb(img)
    0.567
    """  # NOQA: E501
    assert img.mode == "RGB", "Image must be in RGB mode"

    img_arr = np.array(img)

    # First we initialize 3 arrays
    rs, gs, bs = [], [], []
    for row in img_arr:  # for each
        for pixel in row:  # for each pixelò
            # we append the RGB value to the corrisponding list
            rs.append(int(pixel[0]))
            gs.append(int(pixel[1]))
            bs.append(int(pixel[2]))

    rg = [rs[x] - gs[x] for x in range(0, len(rs))]  # evaluate rg
    yb = [0.5 * (rs[x] + gs[x]) - bs[x] for x in range(0, len(rs))]  # evaluate yb

    # evaluate the std of RGYB
    std_rgyb = np.sqrt((float(np.std(rg)) ** 2) + (float(np.std(yb)) ** 2))
    # evaluate the mean of RGYB
    mean_rgyb = np.sqrt((float(np.mean(rg)) ** 2) + (float(np.mean(yb)) ** 2))

    colorfulness = std_rgyb + 0.3 * mean_rgyb  # compute the colorfulness index
    return colorfulness.item()
