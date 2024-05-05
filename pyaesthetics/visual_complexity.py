import io
from dataclasses import dataclass

from PIL.Image import Image as PilImage

from pyaesthetics.utils import QuadTreeDecomposer


@dataclass
class VisualComplexityOutput(object):
    num_blocks: int
    weight: int


def get_visual_complexity(
    img: PilImage, min_std: int, min_size: int
) -> VisualComplexityOutput:
    assert img.mode == "RGB", f"Image must be in RGB mode but is in {img.mode}"

    def get_num_blocks(img: PilImage, min_std: int, min_size: int) -> int:
        quad_tree = QuadTreeDecomposer(img=img, min_std=min_std, min_size=min_size)
        return len(quad_tree.blocks)

    def get_weight(img: PilImage) -> int:
        img_io = io.BytesIO()
        img.save(img_io, format="PNG")
        img_io.seek(0, 2)
        return img_io.tell()

    return VisualComplexityOutput(
        num_blocks=get_num_blocks(img=img, min_std=min_std, min_size=min_size),
        weight=get_weight(img),
    )
