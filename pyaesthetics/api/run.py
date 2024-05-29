from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel

from pyaesthetics.api.routers import (
    analysis,
    brightness,
    color_detection,
    colorfulness,
    face_detection,
    space_based_decomposition,
    symmetry,
    visual_complexity,
)

app = FastAPI(
    license_info={
        "name": "GPL-3.0",
        "url": "https://www.gnu.org/licenses/gpl-3.0.html",
    }
)


@app.exception_handler(RequestValidationError)
async def handler(request: Request, exc: RequestValidationError):
    print(exc)
    return JSONResponse(content={}, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


routers = [
    analysis.router,
    brightness.router,
    color_detection.router,
    colorfulness.router,
    face_detection.router,
    space_based_decomposition.router,
    symmetry.router,
    visual_complexity.router,
]
for router in routers:
    app.include_router(router)


@app.get("/", include_in_schema=False)
async def docs_redirect() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Health check"])
def health() -> Response:
    return Response(content="OK")
