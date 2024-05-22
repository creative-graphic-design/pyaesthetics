from fastapi import APIRouter, File, UploadFile

from pyaesthetics.api.utils import get_image_from_upload_file
from pyaesthetics.color_detection import ColorDetectionOutput, get_colors_w3c

router = APIRouter(prefix="/color-detection", tags=["Color detection"])


@router.post(
    "/",
    response_description="Percentage distribution of colors according to the W3C sixteens basic colors",  # NOQA: E501
)
async def get_color_w3c_endpoint(
    image_file: UploadFile = File(..., description="image to analyze, in RGB"),
) -> ColorDetectionOutput:
    """This endpoint is used to get a simplified color palette (W3C siteens basic colors).

    F = 255
    C0 = 192
    80 = 128
    """
    image = await get_image_from_upload_file(image_file)
    return get_colors_w3c(image)
