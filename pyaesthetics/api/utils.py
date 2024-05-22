import io

from fastapi import UploadFile
from PIL import Image

from pyaesthetics.utils.typehint import PilImage


async def get_image_from_upload_file(upload_file: UploadFile) -> PilImage:
    data = await upload_file.read()
    return Image.open(io.BytesIO(data))
