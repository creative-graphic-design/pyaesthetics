from fastapi import APIRouter, File, UploadFile

from pyaesthetics.api.utils import get_image_from_upload_file
from pyaesthetics.space_based_decomposition import (
    AreasOutput,
    TextImageRatioOutput,
    get_areas,
    get_text_image_ratio,
)
from pyaesthetics.utils.text import TesseractTextDetector, TextDetector

router = APIRouter(prefix="/space-based-decomposition", tags=["Space-based decomposition"])


async def get_areas_async(
    image_file: UploadFile,
    text_detector: TextDetector,
    min_area: int = 100,
    is_resize: bool = True,
    resized_w: int = 600,
    resized_h: int = 400,
    is_plot: bool = False,
    is_coordinates: bool = False,
    is_areatype: bool = True,
) -> AreasOutput:
    image = await get_image_from_upload_file(image_file)
    return get_areas(
        image,
        min_area=min_area,
        is_resize=is_resize,
        resized_w=resized_w,
        resized_h=resized_h,
        is_plot=is_plot,
        is_coordinates=is_coordinates,
        is_areatype=is_areatype,
        text_detector=text_detector,
    )


@router.post("/areas")
@router.post("/areas/tesseract")
async def get_areas_endpoint(
    image_file: UploadFile = File(..., description="image to analyze, in RGB"),
    min_area: int = 100,
    is_resize: bool = True,
    resized_w: int = 600,
    resized_h: int = 400,
    is_plot: bool = False,
    is_coordinates: bool = False,
    is_areatype: bool = True,
) -> AreasOutput:
    areas = await get_areas_async(
        image_file=image_file,
        min_area=min_area,
        is_resize=is_resize,
        resized_w=resized_w,
        resized_h=resized_h,
        is_plot=is_plot,
        is_coordinates=is_coordinates,
        is_areatype=is_areatype,
        text_detector=TesseractTextDetector(),
    )
    return areas


@router.post("/text-image-ratio")
@router.post("/text-image-ratio/tesseract")
async def get_text_image_ratio_endpoint(
    image_file: UploadFile,
    min_area: int = 100,
    is_resize: bool = True,
    resized_w: int = 600,
    resized_h: int = 400,
    is_plot: bool = False,
    is_coordinates: bool = False,
    is_areatype: bool = True,
) -> TextImageRatioOutput:
    areas = await get_areas_async(
        image_file=image_file,
        min_area=min_area,
        is_resize=is_resize,
        resized_w=resized_w,
        resized_h=resized_h,
        is_plot=is_plot,
        is_coordinates=is_coordinates,
        is_areatype=is_areatype,
        text_detector=TesseractTextDetector(),
    )
    return get_text_image_ratio(areas)
