import copy
import http
import re

import ddtrace
from ddtrace import tracer
from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from typing import List, Optional, Dict
from pydantic import BaseModel, validator

import json
import logging
import os
import uuid
import time
import requests

from service_library.constants import NT_ACCOUNT_ID, LLM_AS_A_JUDGE_EVALUATOR_TYPE, CUSTOM_EVALUATOR_TYPE, \
    LOCAL_EVAL_RESULTS_TMP_FOLDER, LLM_AS_A_JUDGE_FILE_NAME, NT_ACCOUNT_NAME, EVAL_ACCOUNT_ID
from service_library.handler.config_loader import ConfigLoader
from service_library.handler.database_handler import DatabaseHandler
from service_library.nemo_ms.nemo_datastore import NeMoDataStore
from service_library.run.comparison_container import ComparisonContainer

from service_library.run.regression_container import RegressionRunContainer
from service_library.utils.configuration_helper import get_evaluation_config, get_platform_config, \
    get_evaluation_project_by_id, get_evaluation_config_by_nemo_evaluator_type, get_platform_config_model_from_dict
from service_library.utils.data_helper import update_dict
from service_library.utils.pydantic_helper import is_valid_instance

from service_library.utils.run_helpers import is_timestamp_in_range, get_formatted_datetime

router = APIRouter(
    tags=["Analysis"]
)

logger = logging.getLogger('Analysis Router')


@router.get("/metrics")
async def eval_post_processing(history_ids: Optional[list] = None):
    """
    example:
    ```
    {
      "HistoryIds":
        [
            6273, 6265
        ]
    }
    ```
    """

    results = {}

    for history_id in history_ids:

        evaluation_history_details = DatabaseHandler({'env': 'dev'}).get_evaluation_history_details(
            history_id=history_id
        )
        if evaluation_history_details is None or len(evaluation_history_details) == 0:
            return JSONResponse(status_code=500,
                                content=jsonable_encoder("Failed to connect to Database", by_alias=False))

        results[history_id] = evaluation_history_details[0]

    return JSONResponse(status_code=200, content=jsonable_encoder(results, by_alias=False))
