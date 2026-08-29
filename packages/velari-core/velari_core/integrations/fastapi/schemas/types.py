from __future__ import annotations

from enum     import StrEnum, auto
from pydantic import BaseModel
from typing   import Dict


class HealthStatus(StrEnum):
    HEALTHY   = auto()
    UNHEALTHY = auto()


class HealthResponse(BaseModel):
    status:  HealthStatus
    service: str
    version: str


class ReadinessResponse(BaseModel):
    status:     HealthStatus
    service:    str
    version:    str
    uptime:     int
    components: Dict[str, str]
