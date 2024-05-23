from PIL.Image import Image  # NOQA
from typing import NewType, Annotated


Base64EncodedImage = NewType("Base64EncodedImage", str)
PilImage = Annotated[Image, "Pillow Image"]
