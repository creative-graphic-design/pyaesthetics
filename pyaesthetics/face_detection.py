#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an entrypoint for automatic analysis of a website.

Created on Mon Apr 16 22:40:46 2018

@author: Giulio Gabrieli, Shunsuke Kitada
"""

from typing import List, Optional, Tuple

from PIL.Image import Image as PilImage
from pydantic import BaseModel, ConfigDict

from pyaesthetics.utils.face import Cv2CancadeClassifier, FaceDetector

###############################################################################
#                                                                             #
#                      Quadratic Tree Decomposition                           #
#                                                                             #
###############################################################################


class GetFacesOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bboxes: List[Tuple[int, int, int, int]]
    num_faces: int
    images: Optional[List[PilImage]] = None


def get_faces(
    img: PilImage,
    is_plot: bool = False,
    face_detector: Optional[FaceDetector] = None,
) -> GetFacesOutput:
    """This functions uses CV2 to get the faces in a pciture.

    :param img: image to analyze in RGB
    :type img: numpy.ndarray
    :param plot: whether to plot or not the results
    :type plot: boolean
    """

    face_detector = face_detector or Cv2CancadeClassifier()
    bboxes = face_detector(img)

    images = face_detector.plot_bboxes(img, bboxes) if is_plot else None
    num_faces = len(bboxes)

    return GetFacesOutput(bboxes=bboxes, num_faces=num_faces, images=images)
