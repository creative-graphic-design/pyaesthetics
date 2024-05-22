#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an entrypoint for automatic analysis of a website.

Created on Mon Apr 16 22:40:46 2018

@author: Giulio Gabrieli, Shunsuke Kitada
"""

from typing import List, Optional, Tuple

from pydantic import BaseModel

from pyaesthetics.utils import Cv2CancadeClassifier, FaceDetector, decode_image, encode_image
from pyaesthetics.utils.typehint import Base64EncodedImage, PilImage

###############################################################################
#                                                                             #
#                      Quadratic Tree Decomposition                           #
#                                                                             #
###############################################################################


class GetFacesOutput(BaseModel):
    bboxes: List[Tuple[int, int, int, int]]
    num_faces: int
    encoded_images: Optional[List[Base64EncodedImage]] = None

    @property
    def images(self) -> Optional[List[PilImage]]:
        return (
            [decode_image(encoded_image) for encoded_image in self.encoded_images]
            if self.encoded_images is not None
            else None
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

    encoded_images = [encode_image(image) for image in images] if images is not None else None
    return GetFacesOutput(bboxes=bboxes, num_faces=num_faces, encoded_images=encoded_images)
