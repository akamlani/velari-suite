from __future__ import annotations

import logging

from fastapi                 import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing                  import Callable, List, Optional, Type

from .exceptions import get_exception_handlers
from .config.settings   import ServiceSettings

logger = logging.getLogger(__name__)


class Server(object):
    def __init__(
        self,
        settings:     ServiceSettings,
        lifespan:     Optional[Callable] = None,
        dependencies: Optional[List]     = None,
    ):
        self._svc          = settings
        self._lifespan_fn  = lifespan
        self._dependencies = dependencies or []
        self.app           = self._build()

    @classmethod
    def from_yaml(
        cls,
        path:         str,
        lifespan:     Optional[Callable] = None,
        dependencies: Optional[List]     = None,
    ) -> Server:
        try:
            return cls(ServiceSettings.from_yaml(path), lifespan=lifespan, dependencies=dependencies)
        except FileNotFoundError:
            logger.error(f"server config not found: {path}")
            raise
        except Exception as e:
            logger.error(f"failed to initialize server from {path}: {e}")
            raise

    def _build(self) -> FastAPI:
        try:
            app = FastAPI(
                title              = self._svc.cfg.info.title,
                description        = self._svc.cfg.info.description,
                version            = self._svc.cfg.info.version,
                docs_url           = self._svc.cfg.info.docs_url,
                redoc_url          = self._svc.cfg.info.redoc_url,
                # the registered lifespan function to manage startup/shutdown events
                lifespan           = self._lifespan_fn,
                # register default global exception handlers
                exception_handlers = get_exception_handlers(),
                # dependencies are callables that are executed before each request and can be used to inject shared resources
                dependencies       = self._dependencies,
            )
            self._setup_middleware(app)
            app.state.settings = self._svc
            return app
        except Exception as e:
            logger.error(f"failed to build FastAPI application: {e}")
            raise

    def _setup_middleware(self, app: FastAPI) -> None:
        _cors = self._svc.cfg.cors
        app.add_middleware(
            CORSMiddleware,
            allow_origins     = list(_cors.allow_origins),
            allow_credentials = bool(_cors.allow_credentials),
            allow_methods     = list(_cors.allow_methods),
            allow_headers     = list(_cors.allow_headers),
            expose_headers    = list(_cors.expose_headers),
        )

    def register_router(self, router: APIRouter) -> None:
        self.app.include_router(router)

    def register_exception_handler(self, exc_type: Type[Exception], handler: Callable) -> None:
        self.app.add_exception_handler(exc_type, handler)

    def register_dependency(self, dependency: Callable) -> None:
        self.app.router.dependencies.append(Depends(dependency))

    def run(self, **kwargs) -> None:
        import uvicorn
        try:
            uvicorn.run(
                self.app,
                host = self._svc.cfg.connection.host,
                port = self._svc.cfg.connection.port,
                **kwargs
            )
        except OSError as e:
            logger.error(f"server failed to start on {self._svc.cfg.connection.host}:{self._svc.cfg.connection.port}: {e}")
            raise
        except Exception as e:
            logger.error(f"server error: {e}")
            raise
