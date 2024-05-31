"""
Provides a function to convert images from sRGB color space to linear RGB color space.

The main function in this module, `s_rgb_to_rgb`, takes an image in sRGB color space as input,
and returns the image in linear RGB color space. This conversion is useful for various image processing
tasks, such as color correction, image analysis, and computer vision tasks.

@author: Shunsuke Kitada

Functions
---------
s_rgb_to_rgb(img: np.ndarray) -> np.ndarray
    Convert an sRGB image to linear RGB values.
"""  # NOQA: E501

import numpy as np


def s_rgb_to_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert an sRGB image to linear RGB values.

    This function takes an image in sRGB color space and converts it to linear RGB color space.
    The conversion is performed according to the standard sRGB conversion formula.

    Parameters
    ----------
    img : np.ndarray
        The input image in sRGB color space. The image should be a 3D array where the third dimension
        represents the color channels (red, green, blue). The values are expected to be in the range of 0-255.

    Returns
    -------
    np.ndarray
        The converted image in linear RGB color space. The output is a 3D array with the same shape as the
        input, where the values represent the linear RGB color channels. The values are in the range of 0-1.

    """  # NOQA: E501
    img = img / 255.0
    mask = img <= 0.04045
    img[mask] /= 12.92
    img[~mask] = ((img[~mask] + 0.055) / 1.055) ** 2.4
    return img
