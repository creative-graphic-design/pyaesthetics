from fastapi import APIRouter, File, UploadFile

from pyaesthetics.api.utils import get_image_from_upload_file
from pyaesthetics.saturation import get_saturation

router = APIRouter(prefix="/saturation", tags=["Saturation"])


async def get_saturation_endpoint(
    image_file: UploadFile = File(..., description="image to analyze, in RGB"),
) -> float:
    """This function evaluates the saturation of an image"""
    image = await get_image_from_upload_file(image_file)
    return get_saturation(image)
