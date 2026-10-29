from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes.v1 import router as v1_router

settings = get_settings()

app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    description="Educational health condition prediction prototype API.",
)

# Configure development and production CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Unversioned legacy/root health check kept intact
@app.get("/health", summary="Basic health check")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


# Include versioned API routers
app.include_router(v1_router)
