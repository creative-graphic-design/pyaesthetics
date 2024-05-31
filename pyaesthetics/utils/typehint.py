"""
Provides type aliases for a base64 encoded image and a Pillow Image.

The `Base64EncodedImage` is a type alias for a string that represents a base64 encoded image.
It is used to indicate that a function or method expects or returns a base64 encoded image.

The `PilImage` is a type alias for the `Image` class from the Pillow library.
It is used to indicate that a function or method expects or returns a Pillow Image.

Type Aliases
------------
Base64EncodedImage : Annotated[str, "Base64EncodedImage"]
    A string that represents a base64 encoded image.

PilImage : Annotated[Image, "Pillow Image"]
    A Pillow `Image` object.
"""  # NOQA: E501

from typing import Annotated

from PIL.Image import Image

Base64EncodedImage = Annotated[str, "Base64EncodedImage"]
"""
A type alias for a base64 encoded image.

The `Base64EncodedImage` is a type alias for a string that represents a base64 encoded image. 
It is used to indicate that a function or method expects or returns a base64 encoded image.

Type Alias
----------
Base64EncodedImage : Annotated[str, "Base64EncodedImage"]
    A string that represents a base64 encoded image.
"""

PilImage = Annotated[Image, "Pillow Image"]
"""
A type alias for a Pillow Image.

The `PilImage` is a type alias for the `Image` class from the Pillow library. 
It is used to indicate that a function or method expects or returns a Pillow Image.

Type Alias
----------
PilImage : Annotated[Image, "Pillow Image"]
    A Pillow `Image` object.
"""
