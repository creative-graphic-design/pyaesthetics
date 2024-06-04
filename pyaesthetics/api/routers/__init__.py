from pyaesthetics.api.routers.analysis import router as analysis_router
from pyaesthetics.api.routers.brightness import router as brightness_router
from pyaesthetics.api.routers.color_detection import router as color_detection_router
from pyaesthetics.api.routers.colorfulness import router as colorfulness_router
from pyaesthetics.api.routers.face_detection import router as face_detection_router
from pyaesthetics.api.routers.saturation import router as saturation_router
from pyaesthetics.api.routers.space_based_decomposition import (
    router as space_based_decomposition_router,
)
from pyaesthetics.api.routers.symmetry import router as symmetry_router
from pyaesthetics.api.routers.visual_complexity import (
    router as visual_complexity_router,
)

__all__ = [
    "analysis_router",
    "brightness_router",
    "color_detection_router",
    "colorfulness_router",
    "face_detection_router",
    "saturation_router",
    "space_based_decomposition_router",
    "symmetry_router",
    "visual_complexity_router",
]
