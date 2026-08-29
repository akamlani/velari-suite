from __future__ import annotations

from fastapi   import Request
from .config.settings import ServiceSettings


def get_settings(request: Request) -> ServiceSettings:
    return request.app.state.settings
