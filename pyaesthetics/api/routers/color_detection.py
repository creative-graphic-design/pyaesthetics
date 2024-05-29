from typing import Annotated

from fastapi import APIRouter, File, Query, UploadFile
from pydantic import BaseModel

from pyaesthetics.api.utils import get_image_from_upload_file
from pyaesthetics.color_detection import ColorDetectionOutput, NColorType, get_colors_w3c

router = APIRouter(prefix="/color-detection", tags=["Color detection"])


class ColorDetectionInput(BaseModel):
    n_colors: Annotated[NColorType, Query(description="number of colors to use")] = 16
    is_plot: Annotated[bool, Query(description="indicates wether to plot the colors")] = False


@router.post(
    "/",
    response_description="Percentage distribution of colors according to the W3C sixteens basic colors",  # NOQA: E501
)
async def get_color_w3c_endpoint(
    image_file: UploadFile = File(..., description="image to analyze, in RGB"),
    color_detection_input: ColorDetectionInput = ColorDetectionInput(),
) -> ColorDetectionOutput:
    """This endpoint is used to get a simplified color palette (W3C siteens basic colors).

    F = 255
    C0 = 192
    80 = 128
    """
    image = await get_image_from_upload_file(image_file)
    return get_colors_w3c(
        image,
        n_colors=color_detection_input.n_colors,
        is_plot=color_detection_input.is_plot,
    )
