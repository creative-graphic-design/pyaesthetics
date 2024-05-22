#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an entrypoint for automatic analysis of a website.

Created on Mon Apr 16 22:40:46 2018

@author: Giulio Gabrieli, Shunsuke Kitada
"""

from typing import List, Optional, Tuple

from pydantic import BaseModel, field_validator

from pyaesthetics.utils import Cv2CancadeClassifier, FaceDetector, encode_image
from pyaesthetics.utils.typehint import EncodedImageStr, PilImage

###############################################################################
#                                                                             #
#                      Quadratic Tree Decomposition                           #
#                                                                             #
###############################################################################


class GetFacesOutput(BaseModel):
    bboxes: List[Tuple[int, int, int, int]]
    num_faces: int
    images: Optional[List[EncodedImageStr]] = None

    @field_validator("images")
    @classmethod
    def encode_images(cls, images: Optional[List[PilImage]]) -> Optional[List[EncodedImageStr]]:
        return (
            [encode_image(image) if isinstance(image, PilImage) else image for image in images]
            if images is not None
            else images
        )


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
