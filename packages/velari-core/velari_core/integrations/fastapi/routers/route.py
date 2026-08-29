from __future__ import annotations

import os
import logging

from http               import HTTPStatus
from fastapi            import APIRouter, Depends, Request
from fastapi.responses  import HTMLResponse
from fastapi.templating import Jinja2Templates
from jinja2             import TemplateError, TemplateNotFound
from typing             import Any, Dict

from ....core                    import read_root_dir
from ....core.perf.profiler      import profile_system
from ..deps                      import get_settings
from ..errors                    import ServiceError
from ..exceptions                import ServerException, raise_exception
from ..schemas.types             import HealthResponse, HealthStatus
from ..config.settings           import ServiceSettings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/",
    summary              = "Service landing page",
    description          = "Renders an HTML landing page from a configurable Jinja2 template. Template path and filename are set in `config/services/`",
    response_class       = HTMLResponse,
    response_description = "HTML landing page for the service",
    responses            = {HTTPStatus.INTERNAL_SERVER_ERROR: {"description": ServiceError.Code.INTERNAL_ERROR}},
)
def index(request: Request, settings: ServiceSettings = Depends(get_settings)):
    try:
        return Jinja2Templates(
            directory=os.path.join(read_root_dir(), settings.cfg.templates.directory)
        ).TemplateResponse(
            request = request,
            name    = settings.cfg.templates.filename,
            context = {
                "title":       settings.cfg.info.title,
                "version":     settings.cfg.info.version,
                "description": settings.cfg.info.description,
                "docs":        settings.cfg.info.docs_url,
            },
        )
    except FileNotFoundError as e:
        logger.error(f"template directory not found: {e}")
        raise_exception(ServerException, str(e))
    except TemplateNotFound as e:
        logger.error(f"template not found: {settings.cfg.templates.filename}: {e}")
        raise_exception(ServerException, str(e))
    except TemplateError as e:
        logger.error(f"template rendering error: {e}")
        raise_exception(ServerException, str(e))
    except Exception as e:
        logger.error(f"index endpoint error: {e}")
        raise_exception(ServerException, str(e))


@router.get(
    "/health",
    tags                 = ["health"],
    summary              = "Liveness check",
    description          = "Returns `healthy` if the service process is running. Does not probe any dependencies.",
    response_model       = HealthResponse,
    response_description = "Service is alive",
    responses            = {HTTPStatus.INTERNAL_SERVER_ERROR: {"description": ServiceError.Code.INTERNAL_ERROR}},
)
def health(settings: ServiceSettings = Depends(get_settings)) -> Dict[str, str]:
    try:
        return {
            "status":  HealthStatus.HEALTHY,
            "service": settings.cfg.info.title,
            "version": settings.cfg.info.version,
        }
    except Exception as e:
        logger.error(f"health endpoint error: {e}")
        raise_exception(ServerException, str(e))


@router.get(
    "/system",
    tags                 = ["system"],
    summary              = "System profile",
    description          = "Returns platform, CPU, memory, and disk metrics for the host running the service.",
    response_description = "System profile information",
    responses            = {HTTPStatus.INTERNAL_SERVER_ERROR: {"description": ServiceError.Code.INTERNAL_ERROR}},
)
def system(settings: ServiceSettings = Depends(get_settings)) -> Dict[str, Any]:
    try:
        data = profile_system()
        data["connector"] = {
            "title":   settings.cfg.info.title,
            "version": settings.cfg.info.version,
        }
        return data
    except Exception as e:
        logger.error(f"system endpoint error: {e}")
        raise_exception(ServerException, str(e))
