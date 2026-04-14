"""
FastAPI application entry point.

Run with:
    poetry run uvicorn src.api.main:app --reload

Interactive docs available at:
    http://127.0.0.1:8000/docs   (Swagger UI)
    http://127.0.0.1:8000/redoc  (ReDoc)
"""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from src.api.routes import router

app = FastAPI(
    title="Apogeum pod Lupą",
    description=(
        "Software-in-the-loop simulation API for student rocket apogee prediction. "
        "Provides endpoints to run 1D flight simulations with configurable noise models "
        "and physical parameters. Designed for future integration of EKF estimation "
        "and Monte Carlo uncertainty analysis."
    ),
    version="0.1.0",
    contact={
        "name": "Rocket Team",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0",
    },
)

app.include_router(router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")
