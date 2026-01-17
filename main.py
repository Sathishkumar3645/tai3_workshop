from fastapi import FastAPI
from app.routers import router
from app.core.config import settings
from app.core.logging_config import setup_logging

setup_logging()
app = FastAPI(title=settings.PROJECT_NAME)
app.include_router(router.router)
