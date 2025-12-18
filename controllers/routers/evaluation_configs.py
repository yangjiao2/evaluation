import asyncio
import http
import logging
import uuid
from typing import Any, List

from fastapi import APIRouter, HTTPException
from fastapi import Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from requests import Request
from configs.settings import get_settings
from data_models.api.eval_config import ConfigResponse

from data_models.api.dataset import Dataset, DatasetsResponse, DatasetResponse, DatasetRequest
from data_models.api.run_maker import FlowConfigRequest, DatasetConfig, NVBotEvaluationConfig
from data_models.dataset_handler import DataContentOutputConfig
from nvbot_models.nvbot_platform_schemas.contracts.config_contracts.langchain_config.nv_langchain_config import \
    NVBotPlatformConfig
from service_library.constants import LOCAL_EVAL_RESULTS_TMP_FOLDER_WITH_END_SLASH, NT_ACCOUNT_ID
from service_library.handler.config_loader import ConfigLoader
from service_library.handler.s3_dataset_handler import S3DatasetHandler
from service_library.nemo_ms.nemo_datastore import NeMoDataStore
from service_library.nemo_ms.nemo_service_helper import download_results_to_local_directory, \
    get_dataset_contents
from service_library.utils.run_helpers import create_folder_if_not_exists

logger = logging.getLogger("Evaluation config router")

router = APIRouter(
    tags=["Eval Configs"]
)


@router.get(
    "/evaluation_configs",
    response_model=ConfigResponse,
    response_model_exclude_unset=True,
)
async def get_evaluation_configs(
        schema_name: str
) -> ConfigResponse:
    """
    List evaluation config. 
    """
    try:
        schema = await ConfigLoader().get_evaluation_schema(schema_name)
        return ConfigResponse(evaluation_config=schema.dict())

    except Exception as ex:
        logger.error(f"Error when fetching evaluation config, {ex}")
        return ConfigResponse(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                              details=f"Error when fetching evaluation config: {ex}")


@router.get(
    "/bot_flowconfig",
    response_model=ConfigResponse,
    response_model_exclude_unset=True,
)
async def get_flowconfig(
        system: str,
        model: str,
        env: str,
) -> ConfigResponse:
    """
    List flow config.
    """
    try:
        raw_config = await ConfigLoader(env).get_flow_model_config(FlowConfigRequest(
            System=system,
            Model=model,
            Environment=env,
            UserId=NT_ACCOUNT_ID
        ))
        flow_config = NVBotPlatformConfig.model_validate(raw_config)

        return ConfigResponse(evaluation_config=flow_config.dict())

    except Exception as ex:
        logger.error(f"Error when fetching evaluation config, {ex}")
        return ConfigResponse(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                              details=f"Error when fetching evaluation config: {ex}")


@router.post(
    "/generate_evaluation_configs",
    response_model=dict,
)
async def generate_evaluation_configs(
        system: str,
        model: str,
        env: str,
        email_recipients: List[str],
        dataset_config: DatasetConfig,
) -> ConfigResponse:
    """
    Generate evaluation config.
    """
    try:
        raw_config = await ConfigLoader(env).get_flow_model_config(FlowConfigRequest(
            System=system,
            Model=model,
            Environment=env,
            UserId=NT_ACCOUNT_ID
        ))
        flow_config = NVBotPlatformConfig.model_validate(raw_config)
        if flow_config.FlowConfig.GraphConfig is not None:
            schema = await ConfigLoader().get_evaluation_schema("graph_evaluation_schema")
        else:
            schema = await ConfigLoader().get_evaluation_schema("evaluation_schema")
        schema = NVBotEvaluationConfig.model_validate(schema)
        schema.EvaluationSchema.Notification.EmailRecipients = email_recipients
        schema.RegressionSchema.DatasetConfig = dataset_config
        return schema.dict(exclude_none=True)

    except Exception as ex:
        logger.error(f"Error when fetching evaluation config, {ex}")
        return ConfigResponse(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                              details=f"Error when fetching evaluation config: {ex}")
