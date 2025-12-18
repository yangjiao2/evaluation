from fastapi import Request, status, HTTPException
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging
logger = logging.getLogger("Exception-Handler")


async def custom_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error processing the request: {exc}")
    return JSONResponse(
        {"detail": str(exc)}, status_code=status.HTTP_400_BAD_REQUEST)


async def custom_http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"Error processing the request: {exc}")
    return await http_exception_handler(request, exc)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors_list = exc.errors()
    error_data = []
    for error in errors_list:
        message = {'message': f'{error.get("loc")[1]} field is required'}
        error_data.append(message)
    logger.error(f'Exception handling request: {exc}')

    return JSONResponse(
        {"detail": error_data}, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)
