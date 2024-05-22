from fastapi import APIRouter, File, UploadFile

from pyaesthetics.api.utils import get_image_from_upload_file
from pyaesthetics.brightness import get_relative_luminance_bt601, get_relative_luminance_bt709

router = APIRouter(prefix="/brightness", tags=["Brightness"])


@router.post(
    "/relative-luminance/bt601",
    response_description="Mean brightness based on BT.601 standard.",
)
async def bt601_endpoint(
    image_file: UploadFile = File(..., description="image to analyze, in RGB"),
) -> float:
    """This endpoint evaluates the brightness of an image by mean of Y, where Y is evaluated as:

    Y = 0.7152G + 0.0722B + 0.2126R
    B = mean(Y)
    """
    image = await get_image_from_upload_file(image_file)
    return get_relative_luminance_bt601(image)


@router.post(
    "/relative-luminance/bt709",
    response_description="Mean brightness based on BT.709 standard.",
)
async def bt709_endpoint(
    image_file: UploadFile = File(..., description="image to analyze, in RGB"),
) -> float:
    """This endpoint evaluates the brightness of an image by mean of Y, where Y is evaluated as:

    Y = 0.587G + 0.114B + 0.299R
    B = mean(Y)
    """
    image = await get_image_from_upload_file(image_file)
    return get_relative_luminance_bt709(image)
