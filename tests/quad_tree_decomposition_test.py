import cv2
from PIL import Image

from pyaesthetics.quad_tree_decomposition import QuadTreeDecomposer
from pyaesthetics.utils import PyaestheticsTestCase


class TestQuadTreeDecomposition(PyaestheticsTestCase):
    def test_quad_tree(self, min_std: int = 15, min_size: int = 40):
        sample_image_path = str(self.FIXTURES_ROOT / "sample.jpg")
        img = Image.open(sample_image_path)
        quad_tree = QuadTreeDecomposer(img=img, min_std=min_std, min_size=min_size)
        plot_img = quad_tree.get_plot()
        assert img.size == plot_img.size
