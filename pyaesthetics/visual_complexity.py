"""
Contains tools for computing the visual complexity of an image.

The module defines a class, `VisualComplexityOutput`, and a function, `get_visual_complexity`.

@author: Giulio Gabrieli, Shunsuke Kitada

Class
-----
VisualComplexityOutput
    Represents the output of a visual complexity operation, containing the number of blocks and optionally the image's weight.

Function
--------
get_visual_complexity(img: PilImage, min_std: int, min_size: int, is_weight: bool = False)
    Calculates the visual complexity of an image.

This module makes use of the `QuadTreeDecomposer` from the `pyaesthetics.utils` module, and also uses the `PilImage`
type from the `pyaesthetics.utils.typehint` module.
"""  # NOQA: E501

import io
from typing import Optional

from pydantic import BaseModel

from pyaesthetics.utils import QuadTreeDecomposer
from pyaesthetics.utils.typehint import PilImage


class VisualComplexityOutput(BaseModel):
    """
    A class used to represent the output of a visual complexity operation.

    Attributes
    ----------
    num_blocks : int
        The number of blocks resulting from the operation.
    weight : int, optional
        The weight of the complexity, by default None

    Examples
    --------
    >>> output = VisualComplexityOutput(num_blocks=5, weight=10)
    """

    num_blocks: int
    weight: Optional[int] = None


def get_visual_complexity(
    img: PilImage, min_std: int, min_size: int, is_weight: bool = False
) -> VisualComplexityOutput:
    """
    Calculate the visual complexity of an image.

    Parameters
    ----------
    img : PilImage
        The image to analyze.
    min_std : int
        The standard deviation threshold for subsequent splitting.
    min_size : int
        The size threshold for subsequent splitting, in pixels.
    is_weight : bool, optional
        If True, the function will also calculate the image's weight (size in bytes), by default False.

    Returns
    -------
    VisualComplexityOutput
        An instance of the VisualComplexityOutput class, containing the number of blocks and optionally the image's weight.

    Raises
    ------
    AssertionError
        If the image is not in RGB mode.

    Examples
    --------
    >>> from PIL import Image
    >>> img = Image.open('path_to_your_image.png')
    >>> complexity_output = get_visual_complexity(img, min_std=10, min_size=100, is_weight=True)
    """  # NOQA: E501
    assert img.mode == "RGB", f"Image must be in RGB mode but is in {img.mode}"

    def get_num_blocks(img: PilImage, min_std: int, min_size: int) -> int:
        quad_tree = QuadTreeDecomposer(img=img, min_std=min_std, min_size=min_size)
        return len(quad_tree.blocks)

    def get_weight(img: PilImage) -> int:
        img_io = io.BytesIO()

        # Here it is assumed that non-PNG image formats may be input,
        # but always save in PNG format and calculate their weight (size) to
        # keep the output consistent.
        img.save(img_io, format="PNG")
        # Seek to the end fo the file
        img_io.seek(0, 2)

        return img_io.tell()

    return VisualComplexityOutput(
        num_blocks=get_num_blocks(img=img, min_std=min_std, min_size=min_size),
        weight=get_weight(img) if is_weight else None,
    )
