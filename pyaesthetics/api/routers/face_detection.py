from typing import Annotated

from fastapi import APIRouter, File, Query, UploadFile

from pyaesthetics.api.utils import get_image_from_upload_file
from pyaesthetics.face_detection import GetFacesOutput, get_faces

router = APIRouter(prefix="/face-detection", tags=["Face detection"])


@router.post("/opencv", response_description="Number of faces in the image")
async def get_faces_opencv_endpoint(
    image_file: UploadFile = File(..., description="image to analyze, in RGB"),
    is_plot: Annotated[
        bool, Query(description="indicates whether to plot detection results")
    ] = False,
) -> GetFacesOutput:
    image = await get_image_from_upload_file(image_file)
    return get_faces(image, is_plot=is_plot)
