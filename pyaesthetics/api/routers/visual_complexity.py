from typing import Annotated

from fastapi import APIRouter, File, Query, UploadFile

from pyaesthetics.api.utils import get_image_from_upload_file
from pyaesthetics.visual_complexity import VisualComplexityOutput, get_visual_complexity

router = APIRouter(prefix="/visual-complexity", tags=["Visual complexity"])


@router.post("/", response_description="Visual complexity")
async def visual_complexity_endpoint(
    image_file: UploadFile = File(..., description="image to analyze, in RGB"),
    min_std: Annotated[int, Query(description="Std threshold for subsequent splitting")] = 15,
    min_size: Annotated[
        int, Query(description="Size threshold for subsequent splitting, in pixel")
    ] = 40,
    is_weight: Annotated[
        bool, Query(description="indicates whether to return the image weight")
    ] = False,
) -> VisualComplexityOutput:
    image = await get_image_from_upload_file(image_file)
    return get_visual_complexity(image, min_std=min_std, min_size=min_size, is_weight=is_weight)
