from __future__ import annotations

from enum   import StrEnum, auto, nonmember
from http   import HTTPStatus
from typing import Dict, Optional

from fastapi import HTTPException


class ServiceError(object):
    class Code(StrEnum):
        BAD_REQUEST           = auto()   # 400 — malformed or invalid request
        UNAUTHORIZED          = auto()   # 401 — authentication required
        FORBIDDEN             = auto()   # 403 — authenticated but not permitted
        NOT_FOUND             = auto()   # 404 — resource does not exist
        CONFLICT              = auto()   # 409 — resource already exists / state conflict
        VALIDATION_ERROR      = auto()   # 422 — request body failed schema validation
        RATE_LIMITED          = auto()   # 429 — too many requests
        INTERNAL_ERROR        = auto()   # 500 — unexpected server error
        BAD_GATEWAY           = auto()   # 502 — upstream / transport error
        SERVICE_UNAVAILABLE   = auto()   # 503 — downstream dependency unavailable
        GATEWAY_TIMEOUT       = auto()   # 504 — upstream timed out
        INVALID_PAYLOAD       = auto()   # 400 — structurally invalid request body
        UNSUPPORTED_MEDIA     = auto()   # 415 — wrong or missing content-type
        CONTENT_TOO_LARGE     = auto()   # 413 — payload exceeds size limit

        _STATUS_BY_CODE = nonmember({
            BAD_REQUEST:         HTTPStatus.BAD_REQUEST,
            UNAUTHORIZED:        HTTPStatus.UNAUTHORIZED,
            FORBIDDEN:           HTTPStatus.FORBIDDEN,
            NOT_FOUND:           HTTPStatus.NOT_FOUND,
            CONFLICT:            HTTPStatus.CONFLICT,
            VALIDATION_ERROR:    HTTPStatus.UNPROCESSABLE_ENTITY,
            RATE_LIMITED:        HTTPStatus.TOO_MANY_REQUESTS,
            INTERNAL_ERROR:      HTTPStatus.INTERNAL_SERVER_ERROR,
            BAD_GATEWAY:         HTTPStatus.BAD_GATEWAY,
            SERVICE_UNAVAILABLE: HTTPStatus.SERVICE_UNAVAILABLE,
            GATEWAY_TIMEOUT:     HTTPStatus.GATEWAY_TIMEOUT,
            INVALID_PAYLOAD:     HTTPStatus.BAD_REQUEST,
            UNSUPPORTED_MEDIA:   HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            CONTENT_TOO_LARGE:   HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        })

    @classmethod
    def detail(cls, code: ServiceError.Code, message: str) -> Dict[str, str]:
        category = getattr(type(code), "_CATEGORY", None)
        result: Dict[str, str] = {}
        if category is not None:
            result["category"] = category
        result["code"]    = code
        result["message"] = message
        return result

    @classmethod
    def exception(cls, code: ServiceError.Code, message: str, status: Optional[HTTPStatus] = None) -> HTTPException:
        return HTTPException(status_code=status or type(code)._STATUS_BY_CODE[code], detail=cls.detail(code, message))
