from typing import Annotated

from fastapi import APIRouter, File, Query, UploadFile

from pyaesthetics.api.utils import get_image_from_upload_file
from pyaesthetics.symmetry import SymmetryOutput, get_symmetry

router = APIRouter(prefix="/symmetry", tags=["Symmetry"])


@router.post("/", response_description="degree of vertical symmetry")
async def get_symmetry_endpoint(
    image_file: UploadFile = File(..., description="image to analyze, in RGB"),
    min_std: Annotated[int, Query(description="Std threshold for subsequent splitting")] = 5,
    min_size: Annotated[
        int, Query(description="Size threshold for subsequent splitting, in pixel")
    ] = 20,
    is_plot: Annotated[bool, Query(description="indicates whether to plot the result")] = False,
) -> SymmetryOutput:
    """This function returns the degree of symmetry (0-100) between the left and right side of an image"""  # NOQA: E501
    image = await get_image_from_upload_file(image_file)
    return get_symmetry(image, min_std=min_std, min_size=min_size, is_plot=is_plot)
