"""
Provides functions for evaluating the presence of different colors in an image.

The module includes functions to get a simplified color palette based on W3C colors, either using 16 basic colors
or 140 extended colors. It also provides a Pydantic model class to represent the output of a color detection operation,
and a function to plot a color palette image.

@author: Giulio Gabrieli, Shunsuke Kitada

Classes
-------
ColorDetectionOutput : BaseModel
    A Pydantic model class that represents the output of a color detection operation.

Functions
---------
get_color_names(ncolors: NColorType) -> Dict[str, Tuple[int, int, int]]
    Retrieve a dictionary of color names and their corresponding RGB values based on the provided `ncolors`.
map_indices_to_color_names(closest_color_indices, colors) -> List[str]
    Map the indices of closest colors to their corresponding color names.
get_colors_w3c(
    img: PilImage,
    n_colors: NColorType = 16,
    is_plot: bool = False,
    plotncolors: int = 5,
) -> ColorDetectionOutput
    Get a simplified color palette (W3C colors) from an image.
"""  # NOQA: E501

import io
from typing import Dict, Final, List, Literal, Optional, Tuple

import numpy as np
from PIL import Image
from pydantic import BaseModel

from pyaesthetics.utils import decode_image, encode_image
from pyaesthetics.utils.typehint import Base64EncodedImage, PilImage

NColorType = Literal[16, 140]
"""
This type hint indicates that the variable can only be one of the specific literal values: 16 or 140.

It is used to restrict the value of a variable to be one of the specified literal or constant values.

Parameters
----------
Literal[16, 140] : typing.Literal
    The literal values 16 and 140.

Examples
--------
>>> def function(color: NColorType):
...     pass
>>> function(16)  # This is valid
>>> function(140)  # This is also valid
>>> function(15)  # This will raise a type error
"""  # NOQA: E501

COLORS: Final[Dict[int, Dict[str, Tuple[int, int, int]]]] = {
    16: {
        "Aqua": (0, 255, 255),
        "Black": (0, 0, 0),
        "Blue": (0, 0, 255),
        "Fuchsia": (255, 0, 255),
        "Gray": (128, 128, 128),
        "Green": (0, 128, 0),
        "Lime": (0, 255, 0),
        "Maroon": (128, 0, 0),
        "Navy": (0, 0, 128),
        "Olive": (128, 128, 0),
        "Purple": (128, 0, 128),
        "Red": (255, 0, 0),
        "Silver": (192, 192, 192),
        "Teal": (0, 128, 128),
        "White": (255, 255, 255),
        "Yellow": (255, 255, 0),
    },
    140: {
        "AliceBlue": (240, 248, 255),
        "AntiqueWhite": (250, 235, 215),
        "Aqua": (0, 255, 255),
        "AquaMarine": (127, 255, 212),
        "Azure": (240, 255, 255),
        "Beige": (245, 245, 220),
        "Bisque": (255, 228, 196),
        "Black": (0, 0, 0),
        "BlanchedAlmond": (255, 235, 205),
        "Blue": (0, 0, 255),
        "BlueViolet": (138, 43, 226),
        "Brown": (165, 42, 42),
        "BurlyWood": (222, 184, 135),
        "CadetBlue": (95, 158, 160),
        "Chartreuse": (127, 255, 0),
        "Chocolate": (210, 105, 30),
        "Coral": (255, 127, 80),
        "CornFlowerBlue": (100, 149, 237),
        "Cornsilk": (255, 248, 220),
        "Crimson": (220, 20, 60),
        "Cyan": (0, 255, 255),
        "DarkBlue": (0, 0, 139),
        "DarkCyan": (0, 139, 139),
        "DarkGoldenRod": (184, 134, 11),
        "DarkGray": (169, 169, 169),
        "DarkGreen": (0, 100, 0),
        "DarkKhaki": (189, 183, 107),
        "DarkMagenta": (139, 0, 139),
        "DarkOliveGreen": (85, 107, 47),
        "DarkOrange": (255, 140, 0),
        "DarkOrchid": (153, 50, 204),
        "DarkRed": (139, 0, 0),
        "DarkSalmon": (233, 150, 122),
        "DarkSeaGreen": (143, 188, 143),
        "DarkSlateBlue": (72, 61, 139),
        "DarkSlateGray": (47, 79, 79),
        "DarkTurquoise": (0, 206, 209),
        "DarkViolet": (148, 0, 211),
        "DeepPink": (255, 20, 147),
        "DeepSkyBlue": (0, 191, 255),
        "DimGray": (105, 105, 105),
        "DodgerBlue": (30, 144, 255),
        "FireBrick": (178, 34, 34),
        "FloralWhite": (255, 250, 240),
        "ForestGreen": (34, 139, 34),
        "Fuchsia": (255, 0, 255),
        "Gainsboro": (220, 220, 220),
        "GhostWhite": (248, 248, 255),
        "Gold": (255, 215, 0),
        "GoldenRod": (218, 165, 32),
        "Gray": (128, 128, 128),
        "Green": (0, 128, 0),
        "GreenYellow": (173, 255, 47),
        "HoneyDew": (240, 255, 240),
        "HotPink": (255, 105, 180),
        "IndianRed": (205, 92, 92),
        "Indigo": (75, 0, 130),
        "Ivory": (255, 255, 240),
        "Khaki": (240, 230, 140),
        "Lavender": (230, 230, 250),
        "LavenderBlush": (255, 240, 245),
        "LawnGreen": (124, 252, 0),
        "LemonChiffon": (255, 250, 205),
        "LightBlue": (173, 216, 230),
        "LightCoral": (240, 128, 128),
        "LightCyan": (224, 255, 255),
        "LightGoldenrodYellow": (250, 250, 210),
        "LightGray": (211, 211, 211),
        "LightGreen": (144, 238, 144),
        "LightPink": (255, 182, 193),
        "LightSalmon": (255, 160, 122),
        "LightSeaGreen": (32, 178, 170),
        "LightSkyBlue": (135, 206, 250),
        "LightSlateGray": (119, 136, 153),
        "LightSteelBlue": (176, 196, 222),
        "LightYellow": (255, 255, 224),
        "Lime": (0, 255, 0),
        "LimeGreen": (50, 205, 50),
        "Linen": (250, 240, 230),
        "Magenta": (255, 0, 255),
        "Maroon": (128, 0, 0),
        "MediumAquaMarine": (102, 205, 170),
        "MediumBlue": (0, 0, 205),
        "MediumOrchid": (186, 85, 211),
        "MediumPurple": (147, 112, 219),
        "MediumSeaGreen": (60, 179, 113),
        "MediumSlateBlue": (123, 104, 238),
        "MediumSpringGreen": (0, 250, 154),
        "MediumTurquoise": (72, 209, 204),
        "MediumVioletRed": (199, 21, 133),
        "MidnightBlue": (25, 25, 112),
        "MintCream": (245, 255, 250),
        "MistyRose": (255, 228, 225),
        "Moccasin": (255, 228, 181),
        "NavajoWhite": (255, 222, 173),
        "Navy": (0, 0, 128),
        "OldLace": (253, 245, 230),
        "Olive": (128, 128, 0),
        "OliveDrab": (107, 142, 35),
        "Orange": (255, 165, 0),
        "OrangeRed": (255, 69, 0),
        "Orchid": (218, 112, 214),
        "PaleGoldenRod": (238, 232, 170),
        "PaleGreen": (152, 251, 152),
        "PaleTurquoise": (175, 238, 238),
        "PaleVioletRed": (219, 112, 147),
        "PapayaWhip": (255, 239, 213),
        "PeachPuff": (255, 218, 185),
        "Peru": (205, 133, 63),
        "Pink": (255, 192, 203),
        "Plum": (221, 160, 221),
        "PowderBlue": (176, 224, 230),
        "Purple": (128, 0, 128),
        "Red": (255, 0, 0),
        "RosyBrown": (188, 143, 143),
        "RoyalBlue": (65, 105, 225),
        "SaddleBrown": (139, 69, 19),
        "Salmon": (250, 128, 114),
        "SandyBrown": (244, 164, 96),
        "SeaGreen": (46, 139, 87),
        "SeaShell": (255, 245, 238),
        "Sienna": (160, 82, 45),
        "Silver": (192, 192, 192),
        "SkyBlue": (135, 206, 235),
        "SlateBlue": (106, 90, 205),
        "SlateGray": (112, 128, 144),
        "Snow": (255, 250, 250),
        "SpringGreen": (0, 255, 127),
        "SteelBlue": (70, 130, 180),
        "Tan": (210, 180, 140),
        "Teal": (0, 128, 128),
        "Thistle": (216, 191, 216),
        "Tomato": (255, 99, 71),
        "Turquoise": (64, 224, 208),
        "Violet": (238, 130, 238),
        "Wheat": (245, 222, 179),
        "White": (255, 255, 255),
        "WhiteSmoke": (245, 245, 245),
        "Yellow": (255, 255, 0),
        "YellowGreen": (154, 205, 50),
    },
}
"""
A dictionary that maps color names to their corresponding RGB values.

The dictionary contains two main keys: 16 and 140. Each key maps to a dictionary that contains color names 
as keys and RGB values as values. The key 16 represents a set of 16 basic colors, and the key 140 represents 
a set of 140 extended colors.

Parameters
----------
Final[Dict[int, Dict[str, Tuple[int, int, int]]]] : typing.Final, typing.Dict
    A dictionary that maps color names to their corresponding RGB values.

Examples
--------
>>> COLORS[16]["Aqua"]
(0, 255, 255)
>>> COLORS[140]["AliceBlue"]
(240, 248, 255)
"""  # NOQA: E501


class ColorDetectionOutput(BaseModel):
    """
    A Pydantic model class that represents the output of a color detection operation.

    Attributes
    ----------
    color_scheme : Dict[str, float]
        A dictionary mapping color names to their corresponding proportions in the image.

    encoded_image : Optional[Base64EncodedImage], default is None
        The base64 encoded image. If provided, it can be decoded into a PIL image using the `image` property.

    Properties
    ----------
    image : Optional[PilImage]
        A property that decodes `encoded_image` into a PIL image. If `encoded_image` is None, this property
        will also be None.

    Examples
    --------
    >>> output = ColorDetectionOutput(color_scheme={"Red": 0.2, "Blue": 0.8}, encoded_image=encoded_image)
    >>> output.color_scheme
    {"Red": 0.2, "Blue": 0.8}
    >>> output.image
    <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x500 at 0x7F1E7D9408E0>
    """  # NOQA: E501

    color_scheme: Dict[str, float]
    encoded_image: Optional[Base64EncodedImage] = None

    @property
    def image(self) -> Optional[PilImage]:
        """
        A property that decodes `encoded_image` into a PIL image.

        This property uses the `decode_image` function to decode the `encoded_image` attribute into a PIL image.
        If `encoded_image` is None, this property will also be None.

        Returns
        -------
        Optional[PilImage]
            A PIL image if `encoded_image` is not None, otherwise None.

        Examples
        --------
        >>> output = ColorDetectionOutput(color_scheme={"Red": 0.2, "Blue": 0.8}, encoded_image=encoded_image)
        >>> output.image
        <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x500 at 0x7F1E7D9408E0>
        """  # NOQA: E501
        return decode_image(self.encoded_image) if self.encoded_image else None


def get_color_names(ncolors: NColorType) -> Dict[str, Tuple[int, int, int]]:
    """
    Retrieve a dictionary of color names and their corresponding RGB values based on the provided `ncolors`.

    The function uses the `COLORS` dictionary and returns the sub-dictionary corresponding to the provided `ncolors`.
    If `ncolors` is 16, it returns a dictionary of 16 basic colors. If `ncolors` is 140, it returns a dictionary of
    140 extended colors.

    Parameters
    ----------
    ncolors : NColorType
        The number of colors. It must be either 16 or 140.

    Returns
    -------
    Dict[str, Tuple[int, int, int]]
        A dictionary mapping color names to their corresponding RGB values.

    Raises
    ------
    ValueError
        If `ncolors` is not 16 or 140.

    Examples
    --------
    >>> get_color_names(16)
    {"Aqua": (0, 255, 255), "Black": (0, 0, 0), ...}

    >>> get_color_names(140)
    {"AliceBlue": (240, 248, 255), "AntiqueWhite": (250, 235, 215), ...}
    """  # NOQA: E501
    try:
        return COLORS[ncolors]
    except KeyError:
        raise ValueError("Invalid value for 'ncolors'. Value must be 16 or 140.")


def map_indices_to_color_names(closest_color_indices, colors) -> List[str]:
    """
    Map the indices of closest colors to their corresponding color names.

    This function takes a list of indices representing the closest colors and a dictionary of colors,
    and returns a list of color names corresponding to these indices.

    Parameters
    ----------
    closest_color_indices : iterable
        An iterable of indices representing the closest colors. Each index corresponds to a color in the `colors` dictionary.

    colors : dict
        A dictionary mapping color names to their corresponding RGB values.

    Returns
    -------
    List[str]
        A list of color names corresponding to the indices in `closest_color_indices`.

    Examples
    --------
    >>> closest_color_indices = [0, 2, 1]
    >>> colors = {"Red": (255, 0, 0), "Green": (0, 255, 0), "Blue": (0, 0, 255)}
    >>> map_indices_to_color_names(closest_color_indices, colors)
    ["Red", "Blue", "Green"]
    """  # NOQA: E501
    colorscheme = []

    # Map the indices to color names
    for row_indices in closest_color_indices:
        row_colors = [list(colors.keys())[index] for index in row_indices]
        colorscheme.extend(row_colors)

    return colorscheme


def get_colors_w3c(
    img: PilImage,
    n_colors: NColorType = 16,
    is_plot: bool = False,
    plotncolors: int = 5,
) -> ColorDetectionOutput:
    """
    Get a simplified color palette (W3C colors) from an image.

    This function analyzes an image and returns a simplified color palette based on W3C colors.
    It can be used with either 16 basic colors or 140 extended colors.

    Parameters
    ----------
    img : PilImage
        The image to analyze, in RGB.
    n_colors : NColorType, optional
        The number of colors to use. It must be either 16 or 140, by default 16.
    is_plot : bool, optional
        Whether to plot a color palette image, by default False.
    plotncolors : int, optional
        The number of colors to use in the palette image, by default 5.

    Returns
    -------
    ColorDetectionOutput
        An object that represents the color detection output. It contains a `color_scheme` attribute that
        is a dictionary mapping color names to their corresponding proportions in the image, and an `image`
        attribute that is a PIL image if `is_plot` is True, otherwise None.

    Raises
    ------
    ValueError
        If `n_colors` is not 16 or 140.

    Examples
    --------
    >>> img = Image.open('example.jpg')
    >>> output = get_colors_w3c(img, n_colors=16, is_plot=True, plotncolors=5)
    >>> output.color_scheme
    {"Red": 0.2, "Blue": 0.8, ...}
    >>> output.image
    <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x500 at 0x7F1E7D9408E0>
    """  # NOQA: E501
    assert img.mode == "RGB", "Image must be in RGB mode"

    img_arr = np.array(img)

    colors = get_color_names(n_colors)

    colors_array = np.array(list(colors.values()))

    dists = np.sum(np.abs(img_arr[:, :, np.newaxis, :3] - colors_array), axis=3)
    closest_color_indices = np.argmin(dists, axis=2)

    color_names = map_indices_to_color_names(closest_color_indices, colors)

    if img_arr.shape[2] == 4:
        alpha = img_arr[:, :, 3]
        # Exclude completely transparent pixels (alpha == 0) from distance calculation
        mask = alpha > 100
        mask = mask.ravel()
        color_names = np.array(color_names)[mask].tolist()

    unique_colors, counts = np.unique(color_names, return_counts=True)
    colorscheme = {c: count / len(color_names) * 100 for c, count in zip(unique_colors, counts)}

    missingcolors = list(set(colors) - set(unique_colors))
    for color in missingcolors:
        colorscheme[color] = 0.0

    colorscheme = {k: float(v) for k, v in sorted(colorscheme.items())}

    def plot_image(colorscheme: Dict[str, float], plotncolors: int, n_colors: int):
        """
        Plot a color palette image using the top `plotncolors` colors from `colorscheme`.

        This function creates a color palette image with the top `plotncolors` colors from `colorscheme`.
        The colors are sorted in descending order of their proportions in `colorscheme`.

        Parameters
        ----------
        colorscheme : Dict[str, float]
            A dictionary mapping color names to their corresponding proportions.
        plotncolors : int
            The number of colors to use in the palette image.
        n_colors : int
            The total number of colors in `colorscheme`.

        Returns
        -------
        PilImage
            A PIL image of the color palette.

        Examples
        --------
        >>> colorscheme = {"Red": 0.2, "Blue": 0.8}
        >>> plot_image(colorscheme, plotncolors=2, n_colors=16)
        <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x500 at 0x7F1E7D9408E0>
        """  # NOQA: E501
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        sorted_data = sorted(colorscheme.items(), key=lambda x: x[1], reverse=True)

        fig, ax = plt.subplots()
        fig.suptitle(f"Top {plotncolors} colors ({n_colors} colors mode)")

        for i in range(0, plotncolors):
            ax.add_patch(patches.Rectangle((i, 0), 1, 1, facecolor=sorted_data[i][0].lower()))
        ax.set_xlim(0, plotncolors)

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    image = (
        plot_image(
            colorscheme=colorscheme,
            plotncolors=plotncolors,
            n_colors=n_colors,
        )
        if is_plot
        else None
    )
    return ColorDetectionOutput(
        color_scheme=colorscheme,
        encoded_image=encode_image(image) if image else None,
    )
