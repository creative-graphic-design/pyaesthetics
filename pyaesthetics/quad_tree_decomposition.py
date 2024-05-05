"""
This file contains class and functions to perform a Quadratic Tree decomposition
of an image and to visually inspect it.

Created on Mon Apr 16 11:49:45 2018

@author: giulio
"""

from dataclasses import dataclass
from typing import Optional

import cv2  # for image manipulation
import numpy as np
from PIL import ImageDraw
from PIL.Image import Image as PilImage

###############################################################################
#                                                                             #
#                      Quadratic Tree Decomposition                           #
#                                                                             #
###############################################################################
""" Thìs sections handles Quadratic Tree Decomposition. """


@dataclass
class QuadTreeDecomposer(object):
    """This class is used to perfrom a QuadTree decomposition of an image.

    During initialization, QuadTree decomposition is done and result are store in self.blocks as a list containing [x,y,height, width,Std].

    To visualize the results, use get_plot().
    """

    min_std: int
    min_size: int

    img: PilImage
    _img_arr: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        img_arr = np.array(self.img)
        self._img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

    @property
    def img_arr(self) -> np.ndarray:
        assert self._img_arr is not None
        return self._img_arr

    def get_plot(self, edgecolor="red", linewidth=1) -> PilImage:
        blocks = list(self.decompose(img=self.img_arr, x=0, y=0))
        img = self.img.copy()
        draw = ImageDraw.Draw(img)

        for block in blocks:
            xy = (block[0], block[1], block[0] + block[2], block[1] + block[3])
            draw.rectangle(xy=xy, outline=edgecolor, width=linewidth)

        return img

    def decompose(self, img: np.ndarray, x: int, y: int):
        """This function evaluate the mean and std of an image, and decides Whether to perform or not other 2 splits of the leave.

        :param img: img to analyze
        :type img: numpy.ndarray
        :param x: x offset of the leaves to analyze
        :type x: int
        :param Y: Y offset of the leaves to analyze
        :type Y: int
        """

        h, w = img.shape
        std = int(img.std())

        if std >= self.min_std and max(h, w) >= self.min_size:
            if w >= h:
                w2 = int(w / 2)
                img1 = img[0:h, 0:w2]
                img2 = img[0:h, w2:]
                yield from self.decompose(img1, x, y)
                yield from self.decompose(img2, x + w2, y)
            else:
                h2 = int(h / 2)
                img1 = img[0:h2, 0:]
                img2 = img[h2:, 0:]
                yield from self.decompose(img1, x, y)
                yield from self.decompose(img2, x, y + h2)

        yield (x, y, w, h, std)
