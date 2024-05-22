from fastapi import APIRouter, File, UploadFile

from pyaesthetics.api.utils import get_image_from_upload_file
from pyaesthetics.space_based_decomposition import (
    AreasOutput,
    TextImageRatioOutput,
    get_areas,
    get_text_image_ratio,
)

router = APIRouter(prefix="/space-based-decomposition", tags=["Space-based decomposition"])


async def _get_areas(
    image_file: UploadFile,
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
    )


@router.post("/areas")
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
    areas = await _get_areas(
        image_file=image_file,
        min_area=min_area,
        is_resize=is_resize,
        resized_w=resized_w,
        resized_h=resized_h,
        is_plot=is_plot,
        is_coordinates=is_coordinates,
        is_areatype=is_areatype,
    )
    return areas


@router.post("/text-image-ratio")
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
    areas = await _get_areas(
        image_file=image_file,
        min_area=min_area,
        is_resize=is_resize,
        resized_w=resized_w,
        resized_h=resized_h,
        is_plot=is_plot,
        is_coordinates=is_coordinates,
        is_areatype=is_areatype,
    )
    return get_text_image_ratio(areas)
