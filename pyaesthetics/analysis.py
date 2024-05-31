"""
Provides an entry point for the automatic analysis of images using pyaeshtetics.

It defines a set of classes and functions to perform various analyses on an image. These include brightness,
visual complexity, symmetry, colorfulness, contrast, and saturation. It also provides functions for face detection,
color detection, and the calculation of the text to image ratio.

The main function of this module is `analyze_image`, which takes an image and a method of analysis as inputs and
returns an `ImageAnalysisOutput` object containing the results of the analysis.

This module uses the `PilImage` type hint for the input image, and the `AnalyzeMethod` type hint for the method of
analysis. The `AnalyzeMethod` can take two values: "fast" for a quick analysis of the image, and "complete" for a
complete analysis of the image.

@author: Giulio Gabrieli, Shunsuke Kitada

Classes:
---------
ImageAnalysisOutput: A class used to represent the output of an image analysis.

Functions:
----------
analyze_image_fast: A function to perform a quick analysis of an image.
analyze_image_complete: A function to perform a complete analysis of an image.
analyze_image: The main function of this module, which performs an analysis of an image based on the specified method.
"""  # NOQA: E501

from typing import Literal, Optional, get_args

from pydantic import BaseModel

from pyaesthetics.brightness import (
    BrightnessOutput,
    get_relative_luminance_bt601,
    get_relative_luminance_bt709,
)
from pyaesthetics.color_detection import ColorDetectionOutput, get_colors_w3c
from pyaesthetics.colorfulness import (
    ColorfulnessOutput,
    get_colorfulness_hsv,
    get_colorfulness_rgb,
)
from pyaesthetics.contrast import ContrastOutput, contrast_michelson, contrast_rms
from pyaesthetics.face_detection import GetFacesOutput, get_faces
from pyaesthetics.saturation import get_saturation
from pyaesthetics.space_based_decomposition import (
    TextImageRatioOutput,
    get_areas,
    get_text_image_ratio,
)
from pyaesthetics.symmetry import SymmetryOutput, get_symmetry
from pyaesthetics.utils.typehint import PilImage
from pyaesthetics.visual_complexity import VisualComplexityOutput, get_visual_complexity

AnalyzeMethod = Literal["fast", "complete"]
"""
A type hint that specifies the method of image analysis.

This type hint is used to indicate the method of image analysis. It can take two values:
- "fast" for a quick analysis of the image.
- "complete" for a complete analysis of the image.

Attributes
----------
"fast" : str
    Specifies that a quick analysis of the image should be performed.
"complete" : str
    Specifies that a complete analysis of the image should be performed.

Examples
--------
>>> method: AnalyzeMethod = "fast"
"""


class ImageAnalysisOutput(BaseModel):
    """
    A class used to represent the output of an image analysis.

    ...

    Attributes
    ----------
    brightness : BrightnessOutput
        An instance of the BrightnessOutput class representing the brightness of the image.
    visual_complexity : VisualComplexityOutput
        An instance of the VisualComplexityOutput class representing the visual complexity of the image.
    symmetry : SymmetryOutput
        An instance of the SymmetryOutput class representing the symmetry of the image.
    colorfulness : ColorfulnessOutput
        An instance of the ColorfulnessOutput class representing the colorfulness of the image.
    contrast : ContrastOutput
        An instance of the ContrastOutput class representing the contrast of the image.
    saturation : float
        A float value representing the saturation of the image.
    faces : GetFacesOutput, optional
        An instance of the GetFacesOutput class representing the faces detected in the image, if any.
    colors : ColorDetectionOutput, optional
        An instance of the ColorDetectionOutput class representing the colors detected in the image, if any.
    text_image_ratio : TextImageRatioOutput, optional
        An instance of the TextImageRatioOutput class representing the ratio of text to image, if applicable.

    """  # NOQA: E501

    brightness: BrightnessOutput
    visual_complexity: VisualComplexityOutput
    symmetry: SymmetryOutput
    colorfulness: ColorfulnessOutput
    contrast: ContrastOutput
    saturation: float

    faces: Optional[GetFacesOutput] = None
    colors: Optional[ColorDetectionOutput] = None
    text_image_ratio: Optional[TextImageRatioOutput] = None


def analyze_image_fast(
    img: PilImage,
    min_std: int,
    min_size: int,
) -> ImageAnalysisOutput:
    """
    Analyze an image quickly and return an ImageAnalysisOutput object.

    This function calculates various image property values such as brightness, visual complexity, symmetry,
    colorfulness, contrast, and saturation.

    Parameters
    ----------
    img : PilImage
        The input image to be analyzed.
    min_std : int
        The minimum standard deviation for the visual complexity and symmetry calculations.
    min_size : int
        The minimum size for the visual complexity and symmetry calculations.

    Returns
    -------
    ImageAnalysisOutput
        An ImageAnalysisOutput object containing the results of the image analysis.

    Examples
    --------
    >>> img = Image.open("example.jpg")
    >>> analyze_image_fast(img, 10, 20)
    ImageAnalysisOutput(brightness=..., visual_complexity=..., symmetry=..., colorfulness=..., contrast=..., saturation=...)
    """  # NOQA: E501
    brightness = BrightnessOutput(
        bt709=get_relative_luminance_bt709(img),
    )
    visual_complexity = get_visual_complexity(
        img=img,
        min_std=min_std,
        min_size=min_size,
        is_weight=False,
    )
    symmetry = get_symmetry(
        img=img,
        min_std=min_std,
        min_size=min_size,
    )
    colorfulness = ColorfulnessOutput(
        rgb=get_colorfulness_rgb(img),
    )
    contrast = ContrastOutput(
        rms=contrast_rms(img),
    )
    saturation = get_saturation(img)

    return ImageAnalysisOutput(
        brightness=brightness,
        visual_complexity=visual_complexity,
        symmetry=symmetry,
        colorfulness=colorfulness,
        contrast=contrast,
        saturation=saturation,
    )


def analyze_image_complete(
    img: PilImage,
    min_std: int,
    min_size: int,
    is_resize: bool,
    resized_w: int,
    resized_h: int,
) -> ImageAnalysisOutput:
    """
    Perform a complete analysis of an image and return an ImageAnalysisOutput object.

    This function calculates various image property values such as brightness, visual complexity, symmetry,
    colorfulness, contrast, and saturation. It also detects faces, colors, and calculates the text to image ratio.

    Parameters
    ----------
    img : PilImage
        The input image to be analyzed.
    min_std : int
        The minimum standard deviation for the visual complexity and symmetry calculations.
    min_size : int
        The minimum size for the visual complexity and symmetry calculations.
    is_resize : bool
        A flag to indicate if the image should be resized for area calculations.
    resized_w : int
        The width to resize the image to for area calculations.
    resized_h : int
        The height to resize the image to for area calculations.

    Returns
    -------
    ImageAnalysisOutput
        An ImageAnalysisOutput object containing the results of the image analysis.

    Examples
    --------
    >>> img = Image.open("example.jpg")
    >>> analyze_image_complete(img, 10, 20, True, 800, 600)
    ImageAnalysisOutput(brightness=..., visual_complexity=..., symmetry=..., colorfulness=..., contrast=..., saturation=..., faces=..., colors=..., text_image_ratio=...)
    """  # NOQA: E501
    brightness = BrightnessOutput(
        bt709=get_relative_luminance_bt709(img),
        bt601=get_relative_luminance_bt601(img),
    )
    visual_complexity = get_visual_complexity(
        img=img,
        min_std=min_std,
        min_size=min_size,
        is_weight=True,
    )
    symmetry = get_symmetry(
        img=img,
        min_std=min_std,
        min_size=min_size,
    )
    colorfulness = ColorfulnessOutput(
        rgb=get_colorfulness_rgb(img),
        hsv=get_colorfulness_hsv(img),
    )
    contrast = ContrastOutput(
        rms=contrast_rms(img),
        michelson=contrast_michelson(img),
    )
    saturation = get_saturation(img)

    faces = get_faces(img=img)
    colors = get_colors_w3c(img=img, n_colors=140)

    areas = get_areas(
        img,
        is_resize=is_resize,
        resized_w=resized_w,
        resized_h=resized_h,
        is_areatype=True,
    )
    text_image_ratio = get_text_image_ratio(areas)

    return ImageAnalysisOutput(
        brightness=brightness,
        visual_complexity=visual_complexity,
        symmetry=symmetry,
        colorfulness=colorfulness,
        contrast=contrast,
        saturation=saturation,
        faces=faces,
        colors=colors,
        text_image_ratio=text_image_ratio,
    )


def analyze_image(
    img: PilImage,
    method: AnalyzeMethod = "fast",
    is_resize: bool = True,
    resized_w: int = 600,
    resized_h: int = 400,
    min_std: int = 10,
    min_size: int = 20,
) -> ImageAnalysisOutput:
    """
    Entry point for the automatic analysis of an image's aesthetic features.

    Depending on the method chosen, this function will perform either a 'fast' or 'complete' analysis of the image.

    Parameters
    ----------
    img : PilImage
        The input image to be analyzed.
    method : AnalyzeMethod, optional
        The method of analysis to use. Valid methods are 'fast' and 'complete'. Default is 'fast'.
    is_resize : bool, optional
        A flag to indicate if the image should be resized to reduce computational workload. Default is True.
    resized_w : int, optional
        The width to resize the image to if is_resize is True. Default is 600.
    resized_h : int, optional
        The height to resize the image to if is_resize is True. Default is 400.
    min_std : int, optional
        The minimum standard deviation for the Quadratic Tree Decomposition. Default is 10.
    min_size : int, optional
        The minimum size for the Quadratic Tree Decomposition. Default is 20.

    Returns
    -------
    ImageAnalysisOutput
        An ImageAnalysisOutput object containing the results of the image analysis.

    Raises
    ------
    ValueError
        If an invalid method is provided.

    Examples
    --------
    >>> img = Image.open("example.jpg")
    >>> analyze_image(img, method="complete", is_resize=True, resized_w=800, resized_h=600, min_std=10, min_size=20)
    ImageAnalysisOutput(brightness=..., visual_complexity=..., symmetry=..., colorfulness=..., contrast=..., saturation=..., faces=..., colors=..., text_image_ratio=...)
    """  # NOQA: E501
    if method == "fast":
        return analyze_image_fast(
            img=img,
            min_std=min_std,
            min_size=min_size,
        )
    elif method == "complete":
        return analyze_image_complete(
            img=img,
            min_std=min_std,
            min_size=min_size,
            is_resize=is_resize,
            resized_w=resized_w,
            resized_h=resized_h,
        )
    else:
        raise ValueError(f"Invalid method {method}. Valid methods are {get_args(AnalyzeMethod)}")
