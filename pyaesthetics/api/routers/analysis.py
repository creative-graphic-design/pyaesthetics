from typing import Annotated

from fastapi import APIRouter, File, Query, UploadFile

from pyaesthetics.analysis import AnalyzeMethod, ImageAnalysisOutput, analyze_image
from pyaesthetics.api.utils import get_image_from_upload_file
from pyaesthetics.utils.text import detect_text

router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.post(
    "/text/tesseract",
    response_description="Number of character in the text",
)
async def tesseract_text_detection_endpoint(
    image_file: UploadFile = File(..., description="image to analyze, in RGB"),
) -> int:
    """This entrypoint uses `pytesseract` to get information about the presence of text in an image."""  # NOQA: E501
    image = await get_image_from_upload_file(upload_file=image_file)
    return detect_text(image)


@router.post(
    "/image",
    response_description="Number of character in the text",
)
async def analyze_image_endpoint(
    image_file: UploadFile = File(..., description="image to analyze, in RGB"),
    method: Annotated[
        AnalyzeMethod,
        Query(
            description="sets to analysis to use. Valid methods are `fast`, `complete`. Default is `complete`."  # NOQA: E501
        ),
    ] = "complete",
    is_resize: Annotated[
        bool,
        Query(
            description="indicates wether to resize the image (reduce computational workload, increase requested time)"  # NOQA: E501
        ),
    ] = True,
    new_size_w: Annotated[
        int,
        Query(
            description="if the image has to be resized, this indicates the new width of the image"
        ),
    ] = 600,
    new_size_h: Annotated[
        int,
        Query(
            description="if the image has to be resized, this indicates the new height of the image"
        ),
    ] = 400,
    min_std: Annotated[
        int,
        Query(description="minimum standard deviation for the Quadratic Tree Decomposition"),
    ] = 10,
    min_size: Annotated[
        int, Query(description="minimum size for the Quadratic Tree Decomposition")
    ] = 20,
) -> ImageAnalysisOutput:
    """This endpoint acts as entrypoint for the automatic analysis of an image aesthetic features."""  # NOQA: E501
    image = await get_image_from_upload_file(upload_file=image_file)

    return analyze_image(
        image,
        method=method,
        is_resize=is_resize,
        resized_w=new_size_w,
        resized_h=new_size_h,
        min_std=min_std,
        min_size=min_size,
    )
