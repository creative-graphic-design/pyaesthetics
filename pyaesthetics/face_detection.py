"""
Provides functions for detecting faces in an image.

The module includes a function to detect faces in an image using a face detector, with the default being
Cv2CancadeClassifier. It also provides a Pydantic model class to represent the output of a face detection operation.

@author: Giulio Gabrieli, Shunsuke Kitada

Classes
-------
GetFacesOutput : BaseModel
    A Pydantic model class that represents the output of a face detection operation.

Functions
---------
get_faces(
    img: PilImage,
    is_plot: bool = False,
    face_detector: Optional[FaceDetector] = None,
) -> GetFacesOutput
    Detect faces in an image using a face detector.
"""  # NOQA: E501

from typing import List, Optional, Tuple

from pydantic import BaseModel

from pyaesthetics.utils import Cv2CancadeClassifier, FaceDetector, decode_image, encode_image
from pyaesthetics.utils.typehint import Base64EncodedImage, PilImage


class GetFacesOutput(BaseModel):
    """
    A Pydantic model class that represents the output of a face detection operation.

    Attributes
    ----------
    bboxes : List[Tuple[int, int, int, int]]
        A list of bounding boxes for the detected faces. Each bounding box is represented by a tuple of four integers.

    num_faces : int
        The number of faces detected.

    encoded_images : Optional[List[Base64EncodedImage]], default is None
        A list of base64 encoded images of the detected faces. If provided, they can be decoded into PIL images
        using the `images` property.

    Properties
    ----------
    images : Optional[List[PilImage]]
        A property that decodes `encoded_images` into a list of PIL images. If `encoded_images` is None, this
        property will also be None.

    Examples
    --------
    >>> output = GetFacesOutput(bboxes=[(10, 10, 50, 50)], num_faces=1, encoded_images=[encoded_image])
    >>> output.bboxes
    [(10, 10, 50, 50)]
    >>> output.num_faces
    1
    >>> output.images
    [<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x500 at 0x7F1E7D9408E0>]
    """  # NOQA: E501

    bboxes: List[Tuple[int, int, int, int]]
    num_faces: int
    encoded_images: Optional[List[Base64EncodedImage]] = None

    @property
    def images(self) -> Optional[List[PilImage]]:
        """
        A property that decodes `encoded_images` into a list of PIL images.

        This property uses the `decode_image` function to decode each image in the `encoded_images` attribute
        into a PIL image. If `encoded_images` is None, this property will also be None.

        Returns
        -------
        Optional[List[PilImage]]
            A list of PIL images if `encoded_images` is not None, otherwise None.

        Examples
        --------
        >>> output = GetFacesOutput(bboxes=[(10, 10, 50, 50)], num_faces=1, encoded_images=[encoded_image])
        >>> output.images
        [<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x500 at 0x7F1E7D9408E0>]
        """  # NOQA: E501
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
    """
    Detect faces in an image using a face detector.

    This function uses a face detector (default is Cv2CancadeClassifier) to detect faces in an image. It returns
    the bounding boxes of the detected faces, the number of faces, and optionally the images of the detected faces.

    Parameters
    ----------
    img : PilImage
        The image to analyze, in RGB.
    is_plot : bool, optional
        Whether to return the images of the detected faces, by default False.
    face_detector : Optional[FaceDetector], optional
        The face detector to use. If not provided, Cv2CancadeClassifier is used by default.

    Returns
    -------
    GetFacesOutput
        An object that represents the face detection output. It contains `bboxes` attribute that is a list of
        bounding boxes for the detected faces, `num_faces` attribute that is the number of faces detected, and
        `images` attribute that is a list of PIL images of the detected faces if `is_plot` is True, otherwise None.

    Examples
    --------
    >>> img = Image.open('example.jpg')
    >>> output = get_faces(img, is_plot=True)
    >>> output.bboxes
    [(10, 10, 50, 50)]
    >>> output.num_faces
    1
    >>> output.images
    [<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x500 at 0x7F1E7D9408E0>]
    """  # NOQA: E501
    face_detector = face_detector or Cv2CancadeClassifier()
    bboxes = face_detector(img)

    images = face_detector.plot_bboxes(img, bboxes) if is_plot else None
    num_faces = len(bboxes)

    encoded_images = [encode_image(image) for image in images] if images is not None else None
    return GetFacesOutput(bboxes=bboxes, num_faces=num_faces, encoded_images=encoded_images)
