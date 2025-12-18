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

from data_models.api.run_maker import RunMakerRequest, NVBotEvaluationConfig, EvaluationProcessingRequest, \
    EvaluationSchema, DatasetConfig, FlowConfigRequest, StorageType

from nvbot_models.nvbot_platform_schemas.contracts.config_contracts.langchain_config.nv_langchain_config import \
    NVBotPlatformConfig
from nvbot_models.nvbot_platform_schemas.contracts.config_contracts.langchain_config.nvbot_langchain_config import \
    NVBotFlowConfig
from nvbot_models.request_models.bot_maker_request import BotMakerRequest

import json
import logging
import os

from service_library.handler.config_loader import ConfigLoader
from service_library.run.cache_upsert_container import CacheUpsertContainer
from service_library.run.nemo_evaluation_container import NemoEvaluationRunContainer

from service_library.run.regression_container import RegressionRunContainer
from service_library.utils.configuration_helper import get_evaluation_config, get_platform_config, \
    get_evaluation_project_by_id, get_evaluation_config_by_nemo_evaluator_type

from service_library.utils.run_helpers import is_timestamp_in_range, get_formatted_datetime

router = APIRouter(
    tags=["Caching Upsert"]
)

log = logging.getLogger('Caching Update Router')

@router.post("/caching_upsert_run")
async def caching_upsert_run(run_request: RunMakerRequest):
    """
    Sample request: \n
    1.
    ```
    {
        "Project": "nvinfo_mixtral_agent_cache_upsert",
        "RunType": "manual",
        "System": "nvinfo",
        "Model": "mixtral_agent",
        "UserId": "nvbot-evaluation",
        "Env": "stg"
    }
    ```
    \n
    """

    # if PlatformConfig is specified, query directly to /chatbot
    # else fetch config from config manager

    project = run_request.Project
    logging.info(f"initiated regression run for {run_request.Project}")
    span = tracer.current_span()
    logging.info(f"tracer id: {span.trace_id if span else 'None'}")
    print(f"Run request: {run_request.dict()}")
    logging.info(f"Run request: {run_request.dict()}")
    env = run_request.Env
    eval_config = run_request.EvaluationConfig
    cuc = None

    bot_config_request = FlowConfigRequest.model_validate(run_request)
    bot_config = await ConfigLoader(run_request.Env).get_bot_config(bot_config_request)
    bot_name = "unknown"
    if bot_config:
        bot_name = bot_config.get("botName")
    results = {"status": "failure", "bot_name": bot_name}

    try:
        ### --- Fetch configs --- ###
        # TODO: fetch platform if PlatformConfig not provided in run_request
        platform_config = run_request.PlatformConfig
        project_id = run_request.ProjectId
        evaluation_project = get_evaluation_project_by_id(run_request.ProjectId)

        nec = None
        # if run_request.PlatformConfig is None:
            # try fetch evaluation project first, if failed, fetch bot config
            # json_platform_config = await get_platform_config(run_request, evaluation_project)
            # assert json_platform_config is not None, f"Failed to fetch bot config for {run_request.Project}, with model {run_request.Model} and system {run_request.System}."
            # run_request.PlatformConfig = NVBotPlatformConfig.model_validate(json_platform_config)

        if run_request.EvaluationConfig is None:
            json_eval_config = await get_evaluation_config(run_request, evaluation_project)
            assert json_eval_config is not None, f"Failed to fetch eval config for {run_request.Project}, with model {run_request.Model} and system {run_request.System}."

        assert json_eval_config is not None, f"Failed to fetch eval config for {run_request.ProjectId}, with {run_request.Project} asset."
        run_request.EvaluationConfig = NVBotEvaluationConfig.model_validate(json_eval_config)

        eval_config = run_request.EvaluationConfig
        regression_schema = eval_config.RegressionSchema
        if run_request.PlatformConfig or (run_request.System and run_request.Model):
            assert regression_schema is not None, f"Failed to find custom eval due to lack of regression schema for {run_request.ProjectId}"

            ### --- Prepare Answer Generation --- ###

            print("⭐Start Cache Upsert run ")
            cuc = CacheUpsertContainer(
                project=project,
                config=eval_config,
                env=env
            )

            dataframe = await cuc.prepare(request=run_request, config=eval_config)
            run_results = await cuc.arun(request=run_request, config=eval_config, df=dataframe)
            regression_result_file = run_results.get("local_filepath")
            results = {**results, "status": "success", "data": run_results.get("data")}
            print ("Local file", regression_result_file)

            # if not eval_config.EvaluationSchema:
            #     results = JSONResponse(status_code=200, content=jsonable_encoder(run_results, by_alias=False))
            # else:
            #     eval_config.EvaluationSchema.NemoEvaluator.DatasetConfig.Engine = StorageType.LOCAL.value
            #     eval_config.EvaluationSchema.NemoEvaluator.DatasetConfig.RunFile = regression_result_file

        ### --- Start Nemo Evaluator Evaluation --- ###
        # print("⭐Start Nemo Evaluator Evaluation")
        # logging.info("⭐Start Nemo Evaluator Evaluation")
        #
        # nec = NemoEvaluationRunContainer(
        #     project=project,
        #     config=eval_config,
        #     env=env,
        # )
        # nec.prepare(request=run_request, config=eval_config)
        # results = await nec.arun(request=run_request, config=eval_config)
        # # results = nec.finish(request=run_request, config=eval_config)
        # # eval_results = nec.finish(request=run_request, config=eval_config, results = results)
        #
        # print("Eval run results", results)
        should_upsert = run_request.RunType.lower() == "upsert"
        if cuc and results and should_upsert:
            cuc.finish(request=run_request, config=eval_config, run_results=results)
    except Exception as ex:
        log.error(f"Error when running cache update run, {ex}")

        return JSONResponse(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                            content=jsonable_encoder(f"Error: {ex}"))

    return JSONResponse(status_code=200, content=jsonable_encoder(results, by_alias=False))

