from __future__ import annotations

from http    import HTTPStatus
from typing  import Any, Callable, Dict, NoReturn, Type

from fastapi            import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses  import JSONResponse

from .errors import ServiceError


### Custom Exceptions
class ApiException(Exception):
    status_code: int               = HTTPStatus.BAD_REQUEST
    code:        ServiceError.Code = ServiceError.Code.BAD_REQUEST

    def __init__(self, message: str):
        self.message = message

class ServerException(ApiException):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    code        = ServiceError.Code.INTERNAL_ERROR

class UnauthorizedException(ApiException):
    status_code = HTTPStatus.UNAUTHORIZED
    code        = ServiceError.Code.UNAUTHORIZED

class ForbiddenException(ApiException):
    status_code = HTTPStatus.FORBIDDEN
    code        = ServiceError.Code.FORBIDDEN

class ServiceException(ApiException):
    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    code        = ServiceError.Code.SERVICE_UNAVAILABLE

class TimeoutException(ApiException):
    status_code = HTTPStatus.GATEWAY_TIMEOUT
    code        = ServiceError.Code.GATEWAY_TIMEOUT

class TransportException(ApiException):
    status_code = HTTPStatus.BAD_GATEWAY
    code        = ServiceError.Code.BAD_GATEWAY

class ContentSizeException(ApiException):
    status_code = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    code        = ServiceError.Code.CONTENT_TOO_LARGE

class ContentException(ApiException):
    status_code = HTTPStatus.UNSUPPORTED_MEDIA_TYPE
    code        = ServiceError.Code.UNSUPPORTED_MEDIA

class InvalidPayloadException(ApiException):
    status_code = HTTPStatus.BAD_REQUEST
    code        = ServiceError.Code.INVALID_PAYLOAD

### exception handler functions
async def api_exception_handler(request: Request, exc: ApiException) -> JSONResponse:
    return JSONResponse(
        status_code = exc.status_code,
        content     = {"code": exc.code, "message": exc.message},
    )


async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        content={"code": ServiceError.Code.VALIDATION_ERROR, "message": str(exc)},
    )


async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        content={"code": ServiceError.Code.INTERNAL_ERROR, "message": str(exc)},
    )

def get_exception_handlers() -> Dict[Any, Callable]:
    return {
        ApiException:           api_exception_handler,
        RequestValidationError: validation_handler,
        Exception:              internal_error_handler,
    }


### Response Utilities
def response_success(data: Any, status_code: HTTPStatus = HTTPStatus.OK) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=data)

def response_error(
    code:        ServiceError.Code,
    message:     str,
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
) -> JSONResponse:
    return JSONResponse(
        status_code = status_code,
        content     = {"code": code, "message": message},
    )

def raise_exception(exc_cls: Type[ApiException], message: str) -> NoReturn:
    raise exc_cls(message)
