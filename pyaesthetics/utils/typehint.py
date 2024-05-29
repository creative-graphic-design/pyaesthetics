from typing import Annotated

from PIL.Image import Image

Base64EncodedImage = Annotated[str, "Base64EncodedImage"]
PilImage = Annotated[Image, "Pillow Image"]
