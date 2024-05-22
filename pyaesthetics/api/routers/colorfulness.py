import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile

from pyaesthetics.api.utils import get_image_from_upload_file
from pyaesthetics.colorfulness import ColorfulnessOutput, get_colorfulness_hsv, get_colorfulness_rgb

router = APIRouter(prefix="/colorfulness", tags=["Colorfulness"])


@router.post("/hsv", response_description="colorfulness index")
async def colorfulness_hsv_endpoint(
    image_file: UploadFile = File(..., description="image to analyze, in RGB"),
) -> float:
    """This function evaluates the colorfulness of a picture using the formula described in Yendrikhovskij et al., 1998.
    Input image is first converted to the HSV color space, then the S values are selected.
    Ci is evaluated with a sum of the mean S and its std, as in:

    Ci = mean(Si)+ std(Si)

    Reference:
    - Yendrikhovskij, S. N., Frans JJ Blommaert, and Huib de Ridder. "Optimizing color reproduction of natural images." Color and Imaging Conference. Vol. 6. Society of Imaging Science and Technology, 1998.
    """  # NOQA: E501
    image = await get_image_from_upload_file(image_file)
    return get_colorfulness_hsv(image)


@router.post("/rgb", response_description="colorfulness index")
async def colorfulness_rgb_endpoint(
    image_file: UploadFile = File(..., description="image to analyze, in RGB"),
) -> float:
    """This function evaluates the colorfulness of a picture using Metric 3 described in Hasler & Suesstrunk, 2003.
    Ci is evaluated with as:

    Ci =std(rgyb) + 0.3 mean(rgyb)   [Equation Y]
    std(rgyb) = sqrt(std(rg)^2+std(yb)^2)
    mean(rgyb) = sqrt(mean(rg)^2+mean(yb)^2)
    rg = R - G
    yb = 0.5(R+G) - B

    Reference:
    - Hasler, David, and Sabine E. Suesstrunk. "Measuring colorfulness in natural images." Human vision and electronic imaging VIII. Vol. 5007. SPIE, 2003.
    """  # NOQA: E501
    image = await get_image_from_upload_file(image_file)
    return get_colorfulness_rgb(image)
