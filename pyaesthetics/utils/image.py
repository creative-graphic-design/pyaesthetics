"""
Provides functions for encoding and decoding images to and from base64 strings.

The `encode_image` function takes a Pillow Image, saves it as a PNG to a BytesIO object, and then encodes
the BytesIO object into a base64 encoded string.

The `decode_image` function takes a base64 encoded string, decodes it into bytes, and then
converts the bytes into a Pillow Image object.

Functions
---------
encode_image(image: PilImage) -> Base64EncodedImage:
    Encode a Pillow Image into a base64 encoded string.

decode_image(encoded_image: Base64EncodedImage) -> PilImage:
    Decode a base64 encoded string into a Pillow Image.

Example
-------
To use this module, import it along with Pillow and pass your image to the `encode_image` function:

    import base64_image_module
    from PIL import Image

    img = Image.new('RGB', (60, 30), color = 'red')
    encoded_img = base64_image_module.encode_image(img)
    decoded_img = base64_image_module.decode_image(encoded_img)
"""  # NOQA: E501

import base64
import io

from PIL import Image

from pyaesthetics.utils.typehint import Base64EncodedImage, PilImage


def encode_image(image: PilImage) -> Base64EncodedImage:
    """
    Encode a Pillow Image into a base64 encoded string.

    This function takes a Pillow Image, saves it as a PNG to a BytesIO object, and then encodes
    the BytesIO object into a base64 encoded string.

    Parameters
    ----------
    image : PilImage
        The input image as a Pillow Image object.

    Returns
    -------
    Base64EncodedImage
        The base64 encoded string of the image.

    Examples
    --------
    >>> from PIL import Image
    >>> img = Image.new('RGB', (60, 30), color = 'red')
    >>> encoded_img = encode_image(img)
    >>> type(encoded_img)
    <class 'str'>
    """
    image_io = io.BytesIO()
    image.save(image_io, format="PNG")
    image_io.seek(0)
    encoded_image = base64.b64encode(image_io.read()).decode("utf-8")
    return Base64EncodedImage(encoded_image)


def decode_image(encoded_image: Base64EncodedImage) -> PilImage:
    """
    Decode a base64 encoded string into a Pillow Image.

    This function takes a base64 encoded string, decodes it into bytes, and then
    converts the bytes into a Pillow Image object.

    Parameters
    ----------
    encoded_image : Base64EncodedImage
        The base64 encoded string of the image.

    Returns
    -------
    PilImage
        The decoded image as a Pillow Image object.

    Examples
    --------
    >>> encoded_img = 'iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=='
    >>> img = decode_image(encoded_img)
    >>> type(img)
    <class 'PIL.PngImagePlugin.PngImageFile'>
    """  # NOQA: E501
    decoded_image = base64.b64decode(encoded_image)
    return Image.open(io.BytesIO(decoded_image))
