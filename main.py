import logging
import logging.config
import os

from pathlib import Path
# from logging_setup import setup_logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from controllers.routers import routes
from controllers.exception_handlers import handlers
# from nvbot_models.middlewares.sanitizerequestmiddleware import SanitizeRequestMiddleware
from ddtrace import config as dd_trace_config
from ddtrace import tracer
from nvbot_utilities.utils.datadog.trace_filter import ErrorFilter

from starlette_context import context
from starlette_context.middleware import ContextMiddleware

dd_trace_config.http_server.error_statuses = '400-599'
tracer.configure(settings={'FILTERS': [ErrorFilter()]})

base_path = Path(__file__).parent
# Initialize the loggers
log_file = Path.joinpath(base_path, 'logging.ini')
logging.config.fileConfig(fname=log_file, disable_existing_loggers=False)

# Initialize the loggers
# if os.environ.get("GITLAB_CI") != "true":
#     log_file = Path.joinpath(base_path, "logging.ini")
# else:
#     log_file = Path.joinpath(base_path, "logging-pipeline.ini")


# setup_logging(log_file)
root_logger = logging.getLogger()

# Initialize the app
app = FastAPI(title='Evaluation',
              description='Integration service to serve evalution responses',
              docs_url='/evaluation/docs',
              version='1.0.0',
              openapi_url='/evaluation/openapi.json')
# Add the routes to app
app.include_router(routes.router)

# Add the middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["*"],
)
app.add_middleware(ContextMiddleware)
# app.add_middleware(SanitizeRequestMiddleware, settings=settings)

# Add exception handlers
app.add_exception_handler(Exception, handlers.custom_exception_handler)
# app.add_exception_handler(HTTPException, handlers.custom_http_exception_handler)


@app.get("/evaluation", description='Health Check Endpoint')
async def status():
    return JSONResponse(
        status_code=200,
        content='Evaluation service is up')
