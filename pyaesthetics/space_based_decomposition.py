"""
Contains functions to compute the number of independent areas in an image. These areas can be of type 'Text' or 'Image'. The module provides functionality to resize images, detect contours, and calculate areas. It also has the ability to detect text within these areas and calculate the ratio of text to image areas.

@author: Giulio Gabrieli, Shunsuke Kitada

Classes
-------
AreaCoordinates : BaseModel
    A Pydantic model class that represents the coordinates of an area.
AreaOutput : BaseModel
    A Pydantic model class that represents the output of an area detection operation.
AreasOutput : BaseModel
    A Pydantic model class that represents the output of multiple area detection operations.
TextImageRatioOutput : BaseModel
    A Pydantic model class that represents the output of a text-to-image ratio calculation.

Constants
---------
AreaType: Literal["Text", "Image"] This type hint indicates that the variable can only be one of the specific literal values: 'Text' or 'Image'.

Functions
---------
get_areas(img: PilImage, min_area: int = 100, is_resize: bool = True, resized_w: int = 600, resized_h: int = 400, is_plot: bool = False, is_coordinates: bool = False, is_areatype: bool = False, text_detector: Optional[TextDetector] = None) -> AreasOutput
    Detect areas in an image and get information about them.
get_text_image_ratio(areas_output: AreasOutput) -> TextImageRatioOutput
    Calculate the text-to-image ratio, the total area of text, the total area of images, and the number of images.
"""  # NOQA: E501

import logging
from typing import List, Literal, Optional, Sequence

import cv2
import numpy as np
from imutils import contours, perspective
from PIL import Image
from pydantic import BaseModel

from pyaesthetics.utils import decode_image, detect_text, encode_image
from pyaesthetics.utils.text import TextDetector
from pyaesthetics.utils.typehint import Base64EncodedImage, PilImage

logger = logging.getLogger(__name__)

AreaType = Literal["Text", "Image"]
"""
This type hint indicates that the variable can only be one of the specific literal values: 'Text' or 'Image'.

It is used to restrict the value of a variable to be one of the specified literal or constant values.

Parameters
----------
Literal["Text", "Image"] : typing.Literal
    The literal values 'Text' and 'Image'.

Examples
--------
>>> def function(area: AreaType):
...     pass
>>> function("Text")  # This is valid
>>> function("Image")  # This is also valid
>>> function("Other")  # This will raise a type error
"""  # NOQA: E501


class AreaCoordinates(BaseModel):
    """
    A Pydantic model class that represents the coordinates of an area.

    Attributes
    ----------
    xmin : int
        The minimum x-coordinate of the area.
    xmax : int
        The maximum x-coordinate of the area.
    ymin : int
        The minimum y-coordinate of the area.
    ymax : int
        The maximum y-coordinate of the area.
    """

    xmin: int
    xmax: int
    ymin: int
    ymax: int


class AreaOutput(BaseModel):
    """
    A Pydantic model class that represents the output of an area detection operation.

    Attributes
    ----------
    area : int
        The area size.
    coordinates : Optional[AreaCoordinates], default is None
        The coordinates of the area. If provided, it should be an instance of the `AreaCoordinates` class.
    area_type : Optional[AreaType], default is None
        The type of the area. It should be either 'Text' or 'Image' if provided.
    """  # NOQA: E501

    area: int
    coordinates: Optional[AreaCoordinates] = None
    area_type: Optional[AreaType] = None


class AreasOutput(BaseModel):
    """
    A Pydantic model class that represents the output of multiple area detection operations.

    Attributes
    ----------
    areas : List[AreaOutput]
        A list of area outputs. Each area output is an instance of the `AreaOutput` class.
    encoded_image : Optional[Base64EncodedImage], default is None
        The base64 encoded image. If provided, it can be decoded into a PIL image using the `image` property.

    Properties
    ----------
    image : Optional[PilImage]
        A property that decodes `encoded_image` into a PIL image. If `encoded_image` is None, this property
        will also be None.
    """  # NOQA: E501

    areas: List[AreaOutput]
    encoded_image: Optional[Base64EncodedImage] = None

    @property
    def images(self) -> Optional[PilImage]:
        """
        A property that decodes `encoded_image` into a PIL image.

        This property uses the `decode_image` function to decode the `encoded_image` attribute into a PIL image.
        If `encoded_image` is None, this property will also be None.

        Returns
        -------
        Optional[PilImage]
            A PIL image if `encoded_image` is not None, otherwise None.
        """  # NOQA: E501
        return decode_image(self.encoded_image) if self.encoded_image is not None else None


class TextImageRatioOutput(BaseModel):
    """
    A Pydantic model class that represents the output of a text-to-image ratio calculation.

    Attributes
    ----------
    text_image_ratio : float
        The ratio of text area to image area.
    text_area : int
        The size of the text area.
    image_area : int
        The size of the image area.
    num_areas : int
        The number of areas detected.
    """  # NOQA: E501

    text_image_ratio: float
    text_area: int
    image_area: int
    num_areas: int


def _get_bboxes_from_contours(
    cnts: Sequence[np.ndarray],
    is_resize: bool,
    original_w: int,
    original_h: int,
    resized_w: int,
    resized_h: int,
) -> List[np.ndarray]:
    """
    Get bounding boxes from contours.

    This function takes a sequence of contours and calculates the bounding box for each contour. If `is_resize` is
    True, the bounding boxes are resized to the size of the original image.

    Parameters
    ----------
    cnts : Sequence[np.ndarray]
        A sequence of contours. Each contour is represented by a numpy array of points.
    is_resize : bool
        Whether to resize the bounding boxes to the size of the original image.
    original_w : int
        The width of the original image.
    original_h : int
        The height of the original image.
    resized_w : int
        The width of the resized image.
    resized_h : int
        The height of the resized image.

    Returns
    -------
    List[np.ndarray]
        A list of bounding boxes. Each bounding box is represented by a numpy array of four corners.

    Examples
    --------
    >>> cnts = [np.array([[10, 10], [10, 50], [50, 50], [50, 10]])]
    >>> _get_bboxes_from_contours(cnts, is_resize=False, original_w=100, original_h=100, resized_w=50, resized_h=50)
    [array([[10, 10], [10, 50], [50, 50], [50, 10]])]
    """  # NOQA: E501
    bboxes = []
    for cnt in cnts:  # for each contour
        min_area_rect = cv2.minAreaRect(cnt)
        min_area_rect_arr = cv2.boxPoints(min_area_rect)
        min_area_rect_arr = min_area_rect_arr.astype("int")
        min_area_rect_arr = perspective.order_points(min_area_rect_arr)
        min_area_rect_arr = min_area_rect_arr.astype("int")

        if is_resize:
            # convert the box to the size of the original image
            min_area_rect_arr = np.array(
                [
                    [
                        int(corner[0] * original_w / resized_w),
                        int(corner[1] * original_h / resized_h),
                    ]
                    for corner in min_area_rect_arr
                ]
            )
        else:
            # convert the box to the size of the original image
            min_area_rect_arr = np.array(
                [[int(corner[0]), int(corner[1])] for corner in min_area_rect_arr]
            )
        bboxes.append(min_area_rect_arr)
    return bboxes


def _plot_contours(img_arr: np.ndarray, bboxes: List[np.ndarray]) -> PilImage:
    """
    Draw contours on an image based on provided bounding boxes.

    This function takes a numpy array representing an image and a list of bounding boxes, and draws a contour
    for each bounding box on the image.

    Parameters
    ----------
    img_arr : np.ndarray
        A numpy array representing an image.
    bboxes : List[np.ndarray]
        A list of bounding boxes. Each bounding box is represented by a numpy array of four corners.

    Returns
    -------
    PilImage
        A PIL image with contours drawn on it.

    Examples
    --------
    >>> img_arr = np.zeros((100, 100, 3), dtype=np.uint8)
    >>> bboxes = [np.array([[10, 10], [10, 50], [50, 50], [50, 10]])]
    >>> _plot_contours(img_arr, bboxes)
    <PIL.Image.Image image mode=RGB size=100x100 at 0x7F1E7D9408E0>
    """  # NOQA: E501
    for bbox in bboxes:
        cv2.drawContours(img_arr, [bbox], -1, (0, 255, 0), 2)
    return Image.fromarray(img_arr)


def _get_area_type(
    is_areatype: bool, imgportion: np.ndarray, text_detector: Optional[TextDetector] = None
) -> Optional[AreaType]:
    """
    Determine the type of an area in an image.

    This function takes a portion of an image and determines whether it is a 'Text' area or an 'Image' area.
    If `is_areatype` is False, it returns None.

    Parameters
    ----------
    is_areatype : bool
        Whether to determine the type of the area.
    imgportion : np.ndarray
        A numpy array representing a portion of an image.
    text_detector : Optional[TextDetector], optional
        The text detector to use. If not provided, a default text detector is used.

    Returns
    -------
    Optional[AreaType]
        The type of the area. It is either 'Text', 'Image', or None.

    Examples
    --------
    >>> imgportion = np.zeros((50, 50, 3), dtype=np.uint8)
    >>> _get_area_type(True, imgportion)
    'Image'
    """  # NOQA: E501
    if not is_areatype:
        return None

    image = Image.fromarray(imgportion)
    num_texts = detect_text(image, text_detector=text_detector)
    return "Text" if num_texts > 0 else "Image"


def _get_area_coordinates(
    is_coordinates: bool, xmin: int, xmax: int, ymin: int, ymax: int
) -> Optional[AreaCoordinates]:
    """
    Get the coordinates of an area.

    This function takes the minimum and maximum x and y coordinates of an area, and returns an `AreaCoordinates`
    object representing these coordinates. If `is_coordinates` is False, it returns None.

    Parameters
    ----------
    is_coordinates : bool
        Whether to get the coordinates of the area.
    xmin : int
        The minimum x-coordinate of the area.
    xmax : int
        The maximum x-coordinate of the area.
    ymin : int
        The minimum y-coordinate of the area.
    ymax : int
        The maximum y-coordinate of the area.

    Returns
    -------
    Optional[AreaCoordinates]
        An `AreaCoordinates` object representing the coordinates of the area, or None.

    Examples
    --------
    >>> _get_area_coordinates(True, 10, 50, 10, 50)
    AreaCoordinates(xmin=10, xmax=50, ymin=10, ymax=50)
    """  # NOQA: E501
    if not is_coordinates:
        return None

    return AreaCoordinates(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)


def get_areas(
    img: PilImage,
    min_area: int = 100,
    is_resize: bool = True,
    resized_w: int = 600,
    resized_h: int = 400,
    is_plot: bool = False,
    is_coordinates: bool = False,
    is_areatype: bool = False,
    text_detector: Optional[TextDetector] = None,
) -> AreasOutput:
    """
    Detect areas in an image and get information about them.

    This function is adapted from https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
    and detects areas in an image, calculates their sizes, and optionally determines their types and
    coordinates. It returns an `AreasOutput` object that contains a list of `AreaOutput` objects representing
    the detected areas, and optionally an image with contours drawn on it.

    The function uses the Canny edge detection and contour detection methods from OpenCV to detect areas. The
    areas are then filtered based on their sizes. If `is_areatype` is True, the function determines whether each
    area is a 'Text' area or an 'Image' area. If `is_coordinates` is True, the function also calculates the
    coordinates of each area.

    Parameters
    ----------
    img : PilImage
        The image to analyze, in RGB.
    min_area : int, optional
        The minimum area size to consider, by default 100.
    is_resize : bool, optional
        Whether to resize the image before detecting areas, by default True.
    resized_w : int, optional
        The width of the resized image, by default 600.
    resized_h : int, optional
        The height of the resized image, by default 400.
    is_plot : bool, optional
        Whether to return an image with contours drawn on it, by default False.
    is_coordinates : bool, optional
        Whether to calculate the coordinates of each area, by default False.
    is_areatype : bool, optional
        Whether to determine the type of each area, by default False.
    text_detector : Optional[TextDetector], optional
        The text detector to use for determining the area type, by default None.

    Returns
    -------
    AreasOutput
        An object that represents the areas detection output. It contains a `areas` attribute that is a list of
        `AreaOutput` objects representing the detected areas, and an `image` attribute that is a PIL image with
        contours drawn on it if `is_plot` is True, otherwise None.

    Examples
    --------
    >>> img = Image.open('example.jpg')
    >>> output = get_areas(img, is_resize=True, is_plot=True, is_coordinates=True, is_areatype=True)
    >>> output.areas
    [AreaOutput(area=2000, coordinates=AreaCoordinates(xmin=10, xmax=50, ymin=10, ymax=50), area_type='Image')]
    >>> output.image
    <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x500 at 0x7F1E7D9408E0>
    """  # NOQA: E501
    assert img.mode == "RGB", f"Image must be in RGB mode but is in {img.mode}"
    img_arr = np.array(img)

    img_original_arr = img_arr.copy()  # source of the image
    oh, ow, _ = img_original_arr.shape  # shape of the orignal image
    assert img.size == (ow, oh)

    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)  # conversion to greyscale

    if is_resize:
        img_arr = cv2.resize(img_arr, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)

    # apply a Gaussina filter
    img_arr = cv2.GaussianBlur(img_arr, ksize=(3, 3), sigmaX=0)
    edged = cv2.Canny(img_arr, threshold1=10, threshold2=100)
    edged = cv2.dilate(edged, kernel=None, iterations=1)  # type: ignore
    edged = cv2.erode(edged, kernel=None, iterations=1)  # type: ignore

    # get the contours
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) < 1:
        logger.warning("No contours found in the image")
        return AreasOutput(areas=[])

    (cnts, _) = contours.sort_contours(cnts)

    bboxes = _get_bboxes_from_contours(
        cnts,
        is_resize=is_resize,
        original_w=ow,
        original_h=oh,
        resized_w=resized_w,
        resized_h=resized_h,
    )
    plot_img = _plot_contours(img_original_arr, bboxes) if is_plot else None  # type: ignore

    """ Now, we can calculate the area of each box, and we can detect if some text is present"""
    areas = []

    for bbox in bboxes:
        t = np.transpose(bbox)
        xmin, xmax = min(t[0]), max(t[0])
        ymin, ymax = min(t[1]), max(t[1])
        area = (xmax - xmin) * (ymax - ymin)

        if area <= min_area:
            continue

        # make sure the coordinates are within the image
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(ow, xmax), min(oh, ymax)

        imgportion = img_original_arr[ymin:ymax, xmin:xmax]
        if imgportion.size == 0:
            continue

        area_type = _get_area_type(
            is_areatype=is_areatype,
            imgportion=imgportion,
            text_detector=text_detector,
        )
        area_coordinates = _get_area_coordinates(
            is_coordinates=is_coordinates, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
        )
        areas.append(
            AreaOutput(
                area=area.item(),
                coordinates=area_coordinates,
                area_type=area_type,
            )
        )

    return AreasOutput(
        areas=areas, encoded_image=encode_image(plot_img) if plot_img is not None else None
    )


def get_text_image_ratio(areas_output: AreasOutput) -> TextImageRatioOutput:
    """
    Calculate the text-to-image ratio, the total area of text, the total area of images, and the number of images.

    This function takes an `AreasOutput` object that contains a list of `AreaOutput` objects representing the
    detected areas, and calculates the text-to-image ratio, the total area of text, the total area of images,
    and the number of images. The text-to-image ratio is defined as the total area of text divided by the sum
    of the total area of text and the total area of images.

    Parameters
    ----------
    areas_output : AreasOutput
        An object that represents the areas detection output. It contains a `areas` attribute that is a list of
        `AreaOutput` objects representing the detected areas.

    Returns
    -------
    TextImageRatioOutput
        An object that represents the text-to-image ratio output. It contains `text_image_ratio` attribute that
        is the text-to-image ratio, `text_area` attribute that is the total area of text, `image_area` attribute
        that is the total area of images, and `num_areas` attribute that is the number of images.

    Examples
    --------
    >>> areas_output = AreasOutput(areas=[AreaOutput(area=2000, area_type='Text'), AreaOutput(area=2000, area_type='Image')])
    >>> get_text_image_ratio(areas_output)
    TextImageRatioOutput(text_image_ratio=0.5, text_area=2000, image_area=2000, num_areas=1)
    """  # NOQA: E501
    image, text = [], []

    for area in areas_output.areas:
        if area.area_type == "Text":
            text.append(area.area)
        elif area.area_type == "Image":
            image.append(area.area)

    # ratio is 0.5 if picture and text occupy the same area, more in more text, less if more images.
    ratio = sum(text) / (sum(image) + sum(text)) if sum(image) + sum(text) > 0 else 0.5

    return TextImageRatioOutput(
        text_image_ratio=ratio,
        text_area=sum(text),
        image_area=sum(image),
        num_areas=len(image),
    )
