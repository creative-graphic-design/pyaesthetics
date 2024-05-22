import base64
import io

from PIL import Image

from pyaesthetics.utils.typehint import EncodedImageStr, PilImage


def encode_image(image: PilImage) -> EncodedImageStr:
    image_io = io.BytesIO()
    image.save(image_io, format="PNG")
    image_io.seek(0)
    encoded_image = base64.b64encode(image_io.read()).decode("utf-8")
    return EncodedImageStr(encoded_image)


def decode_image(encoded_image: EncodedImageStr) -> PilImage:
    decoded_image = base64.b64decode(encoded_image)
    return Image.open(io.BytesIO(decoded_image))
