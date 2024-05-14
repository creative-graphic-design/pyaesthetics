#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an entrypoint for automatic analysis of a website.

Created on Mon Apr 16 22:40:46 2018

@author: Giulio Gabrieli, Shunsuke Kitada
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import ImageDraw
from PIL.Image import Image as PilImage
from pydantic import BaseModel

###############################################################################
#                                                                             #
#                      Quadratic Tree Decomposition                           #
#                                                                             #
###############################################################################


class GetFacesOutput(BaseModel):
    bboxes: List[Tuple[int, int, int, int]]
    num_faces: int
    images: Optional[List[PilImage]] = None


def get_faces(img: PilImage, is_plot: bool = False) -> GetFacesOutput:
    """This functions uses CV2 to get the faces in a pciture.

    :param img: image to analyze in RGB
    :type img: numpy.ndarray
    :param plot: whether to plot or not the results
    :type plot: boolean
    """
    assert img.mode == "RGB", "Image must be in RGB mode"
    img_arr = np.array(img)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    frontalface_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
    )
    faces_bboxes: np.ndarray = frontalface_cascade.detectMultiScale(img_arr, 1.3, 5)  # type: ignore

    if is_plot:
        images = []
        for x, y, w, h in faces_bboxes:
            img_copy = img.copy()
            draw = ImageDraw.Draw(img_copy)
            draw.rectangle((x, y, x + w, y + h), outline="red")
            images.append(img_copy)
    else:
        images = None

    bboxes = (
        faces_bboxes.tolist()
        if isinstance(faces_bboxes, np.ndarray)
        else list(faces_bboxes)  # empty result
    )
    num_faces = len(faces_bboxes)

    return GetFacesOutput(bboxes=bboxes, num_faces=num_faces, images=images)
