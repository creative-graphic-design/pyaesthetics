"""
Provides classes for decomposing an image using a QuadTree decomposition method.

The `DecomposeOutput` class holds the decomposed output which includes the coordinates (x, y) and
the width (w) and height (h) of a bounding box, as well as a standard deviation (std). It also provides
properties to get the left (l), top (t), right (r), and bottom (b) coordinates of the bounding box.

The `QuadTreeDecomposer` class is used to perform a QuadTree decomposition of an image. During
initialization, QuadTree decomposition is done and results are stored as a list containing instances
of `DecomposeOutput` which represent [x, y, w, h , std]. To visualize the results, use `get_plot()`
method.

Classes
-------
DecomposeOutput
    A class to represent the decomposed output.

QuadTreeDecomposer
    A class to perform a QuadTree decomposition of an image.

Example
-------
To use this module, import it along with Pillow and pass your image to the `QuadTreeDecomposer` class:

    from pyaesthetics.utils import QuadTreeDecomposer
    from PIL import Image

    img = Image.new('RGB', (60, 30), color = 'red')
    decomposer = QuadTreeDecomposer(min_std=10, min_size=5, img=img)
    blocks = decomposer.blocks
    img_with_blocks = decomposer.get_plot()
"""  # NOQA: E501

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np
from PIL import ImageDraw
from pydantic import BaseModel

from pyaesthetics.utils.typehint import PilImage


class DecomposeOutput(BaseModel):
    """
    A class to represent the decomposed output.

    This class holds the decomposed output which includes the coordinates (x, y) and
    the width (w) and height (h) of a bounding box, as well as a standard deviation (std).
    It also provides properties to get the left (l), top (t), right (r), and bottom (b)
    coordinates of the bounding box.

    Attributes
    ----------
    x : int
        The x-coordinate of the bounding box.
    y : int
        The y-coordinate of the bounding box.
    w : int
        The width of the bounding box.
    h : int
        The height of the bounding box.
    std : int
        The standard deviation.

    Methods
    -------
    to_coordinates() -> Tuple[int, int, int, int]:
        Returns the coordinates of the bounding box as a tuple.

    Properties
    ----------
    l : int
        The left coordinate of the bounding box.
    t : int
        The top coordinate of the bounding box.
    r : int
        The right coordinate of the bounding box.
    b : int
        The bottom coordinate of the bounding box.
    """

    x: int
    y: int
    w: int
    h: int
    std: int

    @property
    def l(self) -> int:
        """
        The left coordinate of the bounding box.

        Returns
        -------
        int
            The left coordinate of the bounding box.
        """
        return self.x

    @property
    def t(self) -> int:
        """
        The top coordinate of the bounding box.

        Returns
        -------
        int
            The top coordinate of the bounding box.
        """
        return self.y

    @property
    def r(self) -> int:
        """
        The right coordinate of the bounding box.

        Returns
        -------
        int
            The right coordinate of the bounding box.
        """
        return self.x + self.w

    @property
    def b(self) -> int:
        """
        The bottom coordinate of the bounding box.

        Returns
        -------
        int
            The bottom coordinate of the bounding box.
        """
        return self.y + self.h

    def to_coordinates(self) -> Tuple[int, int, int, int]:
        """
        Return the coordinates of the bounding box as a tuple.

        Returns
        -------
        Tuple[int, int, int, int]
            The coordinates of the bounding box.
        """
        return self.x, self.y, self.w, self.h


@dataclass
class QuadTreeDecomposer(object):
    """
    A class to perform a QuadTree decomposition of an image.

    This class is used to perform a QuadTree decomposition of an image. During initialization,
    QuadTree decomposition is done and results are stored in `self.blocks` as a list containing
    instances of `DecomposeOutput` which represent [x, y, w, h , std]. To visualize the results,
    use `get_plot()` method.

    Attributes
    ----------
    min_std : int
        The minimum standard deviation to consider for decomposition.
    min_size : int
        The minimum size of the block to consider for decomposition.
    img : PilImage
        The input image as a Pillow Image object.

    Methods
    -------
    get_plot(edgecolor="red", linewidth=1) -> PilImage:
        Returns the image with the blocks outlined.

    decompose(x: int, y: int) -> List[DecomposeOutput]:
        Decompose the image starting from the given coordinates.

    Properties
    ----------
    img_arr : np.ndarray
        The image as a numpy array.

    blocks : List[DecomposeOutput]
        The blocks of the decomposed image.
    """

    min_std: int
    min_size: int
    img: PilImage

    _img_arr: Optional[np.ndarray] = field(default=None, repr=False)
    _blocks: Optional[List[DecomposeOutput]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """
        Run this special method after the object is fully initialized.

        It asserts that the image is in RGB mode and converts the image to grayscale.

        Raises
        ------
        AssertionError
            If the image is not in RGB mode.
        """
        assert self.img.mode == "RGB", f"Image must be in RGB mode but is in {self.img.mode} mode"
        img_arr = np.array(self.img)
        self._img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

    @property
    def img_arr(self) -> np.ndarray:
        """
        Get the image as a numpy array.

        This property returns the image as a numpy array.
        If the image array is not initialized, it throws an AssertionError.

        Returns
        -------
        np.ndarray
            The image as a numpy array.

        Raises
        ------
        AssertionError
            If the image array is not initialized.
        """
        assert self._img_arr is not None
        return self._img_arr

    @property
    def blocks(self) -> List[DecomposeOutput]:
        """
        Get the blocks of the decomposed image.

        This property returns the blocks of the decomposed image. If the blocks are not
        initialized, it decomposes the image starting from (0, 0) and sets the blocks.

        Returns
        -------
        List[DecomposeOutput]
            The blocks of the decomposed image.
        """
        if self._blocks is None:
            self._blocks = self.decompose(x=0, y=0)
        return self._blocks

    def get_plot(self, edgecolor="red", linewidth=1) -> PilImage:
        """
        Return the image with the blocks outlined.

        Parameters
        ----------
        edgecolor : str, optional
            The color of the outlines of the blocks, default is "red".
        linewidth : int, optional
            The width of the outlines of the blocks, default is 1.

        Returns
        -------
        PilImage
            The image with the blocks outlined.
        """
        blocks = self.blocks
        img = self.img.copy()
        draw = ImageDraw.Draw(img)

        for block in blocks:
            xy = (block.l, block.t, block.r, block.b)
            draw.rectangle(xy=xy, outline=edgecolor, width=linewidth)

        return img

    def decompose(self, x: int, y: int) -> List[DecomposeOutput]:
        """
        Decompose the image starting from the given coordinates.

        Parameters
        ----------
        x : int
            The x-coordinate to start the decomposition from.
        y : int
            The y-coordinate to start the decomposition from.

        Returns
        -------
        List[DecomposeOutput]
            The blocks of the decomposed image.
        """
        return list(self._decompose(self.img_arr, x, y))

    def _decompose(self, img: np.ndarray, x: int, y: int) -> Iterator[DecomposeOutput]:
        """
        Private method to evaluate the mean and std of an image, and decides whether to perform or not other 2 splits of the leave.

        Parameters
        ----------
        img : np.ndarray
            The image to analyze.
        x : int
            The x offset of the leaves to analyze.
        y : int
            The y offset of the leaves to analyze.

        Yields
        ------
        DecomposeOutput
            The decomposed output of the image.
        """  # NOQA: E501
        h, w = img.shape
        std = int(img.std())

        if std >= self.min_std and max(h, w) >= self.min_size:
            if w >= h:
                w2 = int(w / 2)  # get the new width
                img1 = img[0:h, 0:w2]  # create a subimage
                img2 = img[0:h, w2:]  # create the second subimage
                yield from self._decompose(img1, x, y)
                yield from self._decompose(img2, x + w2, y)

            else:
                h2 = int(h / 2)
                img1 = img[0:h2, 0:]
                img2 = img[h2:, 0:]
                yield from self._decompose(img1, x, y)
                yield from self._decompose(img2, x, y + h2)

        yield DecomposeOutput(x=x, y=y, w=w, h=h, std=std)
