"""
Contains tools for computing the degree of symmetry in an image using QuadTree Decomposition.

The module defines two classes, `SymmetryImage` and `SymmetryOutput`, and a function, `get_symmetry`.

@author: Giulio Gabrieli, Shunsuke Kitada

Classes
-------
SymmetryImage
    Represents an image with symmetry and has the ability to save the left and right parts of the symmetric image.
SymmetryOutput
    Represents the output of a symmetry operation, containing the degree of symmetry and optionally the decomposed images.

Functions
---------
get_symmetry(img: PilImage, min_std: int, min_size: int, is_plot: bool = False)
    Calculates the degree of symmetry between the left and right side of an image.

This module makes use of the `QuadTreeDecomposer`, `decode_image`, and `encode_image` functions from the
`pyaesthetics.utils` module, and also uses the `Base64EncodedImage` and `PilImage` types from the
`pyaesthetics.utils.typehint` module.
"""  # NOQA: E501

import os
from typing import Optional

import numpy as np
from PIL import Image
from pydantic import BaseModel

from pyaesthetics.utils import QuadTreeDecomposer, decode_image, encode_image
from pyaesthetics.utils.typehint import Base64EncodedImage, PilImage

###############################################################################
#                                                                             #
#                                  Symmetry                                   #
#                                                                             #
###############################################################################
""" Thìs sections handles Quadratic Tree Decomposition. """


class SymmetryImage(BaseModel):
    """
    A class used to represent an Image with Symmetry.

    Attributes
    ----------
    left : Base64EncodedImage
        The left part of the symmetric image.
    right : Base64EncodedImage
        The right part of the symmetric image.

    Methods
    -------
    save_images(save_dir_path: str = ".")
        Saves the left and right images to the given directory.
    """

    left: Base64EncodedImage
    right: Base64EncodedImage

    def save_images(self, save_dir_path: str = ".") -> None:
        """
        Save the left and right images to the given directory.

        Parameters
        ----------
        save_dir_path : str, optional
            The directory path where the images should be saved, by default "." (current directory)

        Raises
        ------
        IOError
            If the save_dir_path does not exist or is not writable.

        Examples
        --------
        >>> img = SymmetryImage(left=base64_img1, right=base64_img2)
        >>> img.save_images("/path/to/save/images")
        """
        image_l = decode_image(self.left)
        image_r = decode_image(self.right)

        image_l.save(os.path.join(save_dir_path, "left.png"))
        image_r.save(os.path.join(save_dir_path, "right.png"))


class SymmetryOutput(BaseModel):
    """
    A class used to represent the output of a symmetry operation.

    Attributes
    ----------
    degree : float
        The degree of symmetry.
    images : SymmetryImage, optional
        An instance of the SymmetryImage class representing the left and right parts of the symmetric image,
        by default None

    Examples
    --------
    >>> output = SymmetryOutput(degree=90.0, images=symmetry_image_instance)
    """  # NOQA: E501

    degree: float
    images: Optional[SymmetryImage] = None


def get_symmetry(img: PilImage, min_std: int, min_size: int, is_plot: bool = False):
    """
    Calculate the degree of symmetry between the left and right side of an image.

    Parameters
    ----------
    img : PilImage
        The image to analyze.
    min_std : int
        The standard deviation threshold for subsequent splitting.
    min_size : int
        The size threshold for subsequent splitting, in pixels.
    is_plot : bool, optional
        If True, the function will also return the plots of the decomposed images, by default False.

    Returns
    -------
    SymmetryOutput
        An instance of the SymmetryOutput class, containing the degree of symmetry and optionally the decomposed images.

    Raises
    ------
    AssertionError
        If the image is not in RGB mode or if the image size does not match the shape of the image array.

    Examples
    --------
    >>> from PIL import Image
    >>> img = Image.open('path_to_your_image.png')
    >>> symmetry_output = get_symmetry(img, min_std=10, min_size=100, is_plot=True)
    """  # NOQA: E501
    assert img.mode == "RGB", f"Image must be in RGB mode but is in {img.mode}"
    img_arr = np.array(img)

    h, w, _ = img_arr.shape
    assert img.size == (w, h)

    if h % 2 != 0:
        img_arr = img_arr[:-1, :]
    if w % 2 != 0:
        img_arr = img_arr[:, :-1]

    img_arr_l = img_arr[0:, 0 : int(w / 2)]
    img_arr_r = np.flip(img_arr[0:, int(w / 2) :], 1)

    img_l = Image.fromarray(img_arr_l)
    img_r = Image.fromarray(img_arr_r)

    tree_l = QuadTreeDecomposer(img=img_l, min_std=min_std, min_size=min_size)
    tree_r = QuadTreeDecomposer(img=img_r, min_std=min_std, min_size=min_size)

    blocks_l = tree_l.decompose(x=0, y=0)
    blocks_r = tree_r.decompose(x=0, y=0)

    counter = 0
    tot = len(blocks_r) + len(blocks_l)
    for block_r in blocks_r:
        for block_l in blocks_l:
            if block_r.to_coordinates() == block_l.to_coordinates():
                counter += 1
    degree = counter / tot * 200

    images = (
        SymmetryImage(left=encode_image(tree_l.get_plot()), right=encode_image(tree_r.get_plot()))
        if is_plot
        else None
    )

    return SymmetryOutput(degree=degree, images=images)
