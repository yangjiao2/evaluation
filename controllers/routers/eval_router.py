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
from starlette_context import context

from data_models.api.run_maker import RunMakerRequest, NVBotEvaluationConfig, EvaluationProcessingRequest, \
    EvaluationSchema, DatasetConfig, FlowConfigRequest, StorageType, ComparisonSchema, RegressionSchema, \
    BotPlatformConfig
from nvbot_models.nvbot_platform_schemas.contracts.config_contracts.langchain_config.nvbot_langchain_config import \
    NVBotFlowConfig
from nvbot_models.request_models.bot_maker_request import BotMakerRequest

import json
import logging
import os
import uuid
import time
import requests

from nvbot_models.request_models.evaluation_request import EvaluationRunStatus, UserEvaluationRunData
from service_library.constants import NT_ACCOUNT_ID, LLM_AS_A_JUDGE_EVALUATOR_TYPE, CUSTOM_EVALUATOR_TYPE, \
    LOCAL_EVAL_RESULTS_TMP_FOLDER, LLM_AS_A_JUDGE_FILE_NAME, NT_ACCOUNT_NAME, EVAL_ACCOUNT_ID
from service_library.handler.config_loader import ConfigLoader
from service_library.handler.database_handler import DatabaseHandler
from service_library.nemo_ms.nemo_datastore import NeMoDataStore
from service_library.run.comparison_container import ComparisonContainer
from service_library.run.nemo_evaluation_container import NemoEvaluationRunContainer

from service_library.run.regression_container import RegressionRunContainer
from service_library.utils.configuration_helper import get_evaluation_config, get_platform_config, \
    get_evaluation_project_by_id, get_evaluation_config_by_nemo_evaluator_type, get_platform_config_model_from_dict
from service_library.utils.data_helper import update_dict
from service_library.utils.pydantic_helper import is_valid_instance

from service_library.utils.run_helpers import is_timestamp_in_range, get_formatted_datetime

router = APIRouter(
    tags=["Evaluation"]
)

logger = logging.getLogger('Eval Router')


def _convert_filters_format(request_dict: dict):
    filters_dict = {}

    for camel_key, value in request_dict.items():
        snake_key = re.sub('([a-z0-9])([A-Z])', r'\1_\2', camel_key).lower()
        if snake_key in ["created_date_from", "created_date_to", "modified_date_from", "modified_date_to"]:
            if value is not None:
                filters_dict[snake_key] = get_formatted_datetime(None, json.loads(value))
        else:
            filters_dict[snake_key] = value
    return filters_dict


# @router.post("/batch_answer")
# async def batch_answer_generation(run_request: RunMakerRequest):
#     """
#     Answer generation. \n
#     Sample request:
#     {
#         "Project": "avc_mixtral",
#         "RunType": "manual",
#         "System": "avc",
#         "Model": "mixtral",
#         "Attachments": []
#     } or
#     {
#         "Project": "nvbot_for_nvhelp_mixtral_agent",
#         "RunType": "manual",
#         "System": "nvhelp",
#         "Model": "mixtral_agent",
#         "Attachments": []
#     }
#     """
#     project = run_request.Project
#
#     logger.info(f"initiated regression run for {run_request.Project}")
#
#     try:
#         # run_request.QueryId = run_request.QueryId if run_request.QueryId else str(uuid.uuid4())
#
#         # TODO: fetch platform if PlatformConfig not provided in run_request
#         if run_request.PlatformConfig is None:
#             platform_config = await ConfigLoader().get_flow_model_config(run_request)
#             run_request.PlatformConfig = NVBotPlatformConfig.model_validate(platform_config)
#
#         print(f"Fetched flow Config file for {project} ")
#
#         # TODO: migrate to config manager repo
#         eval_config = None
#         with open(f'asset/{project}.json', 'r') as json_file:
#             json_eval_config = json.load(json_file)
#             run_request.EvaluationConfig = NVBotEvaluationConfig.model_validate(json_eval_config)
#
#         regression_config = run_request.EvaluationConfig
#         ### --- Prepare Answer Generation --- ###
#         # regression_config = eval_config.RegressionSchema
#         rc = RegressionRunContainer(
#             project=project,
#             config=regression_config,
#         )
#
#         dataframe = rc.prepare(request=run_request, config=regression_config)
#         results = await rc.arun(request=run_request, config=regression_config, df=dataframe)
#         # TODO: if local run, dont' upload
#         # results = rc.finish(request=run_request, config=regression_config)
#
#
#     except Exception as ex:
#         log.error(f"Error when running regression, {ex}")
#         return JSONResponse(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
#                             content=jsonable_encoder(f"Error: {ex}"))
#
#     return JSONResponse(status_code=200, content=jsonable_encoder(results, by_alias=False))


@router.post("/run")
async def run(run_request: RunMakerRequest):
    """
    Initiate a evaluation run which will be processed in queue.\n
    Sample request: \n
    1.
    ```
    {
        "Project": "nvbot_for_nvhelp_mixtral_agent_complete_evaluation",
        "RunType": "manual",
        "System": "nvhelp",
        "Model": "mixtral_agent",
        "Attachments": [],
        "Env": "dev"
    }
    ```
    \n
    2.
    ```
    {
        "Project": "avc_mixtral_sample_custom_evaluation",
        "RunType": "manual",
        "System": "avc",
        "Model": "mixtral",
        "Attachments": [],
        "Env": "dev"
    }
    ```
    \n
    3.
    ```
    {
        "Project": "nvinfo_llama4_complete_evaluation",
        "RunType": "manual",
        "System": "nvinfo",
        "Model": "llama4",
        "UserId": "nvbot-evaluation",
        "Env": "dev",
    }
    ```
    \n
    4.
    ```
    {
        "Project": "scout_mixtral_complete_evaluation",
        "RunType": "manual",
        "System": "scout_long",
        "Model": "mixtral",
        "Attachments": [],
        "UserId": "nvbot-evaluation",
        "Env": "sandbox"
    }
    ```
    \n
    5.
    ```
    {
        "Project": "orchestrator_perceptor_complete_evaluation",
        "RunType": "manual",
        "System": "orchestrator_perceptor",
        "Model": "llama_3_1",
        "Attachments": [],
        "UserId": "nvbot-evaluation",
        "Env": "sandbox"
    }
    ```
    \n
    6.
    ```
    {
        "Project": "global_protect_complete_evaluation",
        "RunType": "manual",
        "System": "global_protect_bot_glean_collection",
        "Model": "mixtral_agent",
        "UserId": "yangj",
        "Env": "dev"
    }
    ```

    \n
    7.
    ```
    {
        "Project": "developer_knowledge_expert_complete_evaluation",
        "RunType": "manual",
        "System": "developer_knowledge_expert",
        "Model": "llama_3_1_agent_graph",
        "UserId": "nvbot-evaluation",
        "Env": "sandbox",
        "Parameters": {}
    }
    ```
    \n\n\n

    Customization example

    ```
    {
      "Project": "nvinfo_mixtral_agent_complete_evaluation",
      "RunType": "manual",
      "System": "nvinfo",
      "Model": "mixtral_agent",
      "Attachments": [],
      "UserId": "nvbot-evaluation",
      "Env": "sandbox",
      "Customization": {
       "RegressionSchema": {
       "DatasetConfig": {
         "DataLimit": 20
          }
        }
      }
    }
    ```

    ```
    {
     "Project": "nvinfo_mixtral_agent_complete_evaluation",
     "RunType": "manual",
     "System": "nvinfo",
     "Model": "mixtral_agent",
     "Attachments": [],
     "UserId": "nvbot-evaluation",
     "Env": "sandbox",
     "Customization": {
       "RegressionSchema": {
         "DatasetConfig": {
           "Engine": "s3",
           "DatasetFolder": "nvinfo_mixtral_agent",
           "Name": "nvinfo_mixtral_agent",
           "DatasetPath": "dataset/nvinfo_nov21.xlsx"
         }
       },
       "EvaluationSchema": {
         "Notification": {
           "EmailRecipients": []
         }
       }
     }
    }
    ```

    8. va

    ```
    {
        "Project": "va_complete_evaluation",
        "RunType": "manual",
        "System": "va",
        "UserId": "nvbot-evaluation",
        "Env": "stg",
        "PlatformConfig": {
            "ConfigType": "api",
            "URL": "https://nvidiastage.service-now.com/api/now/vaconversationalapis/getResponse",
            "Auth": {
                "AuthType": "basic",
                "Username": "testing.user",
                "Password": "Password.123"
            },
            "Payload": {}
        }
    }
    ```

    9. nvbugs
    ```
    {
        "Project": "nvbugs_complete_evaluation",
        "RunType": "manual",
        "System": "bugnemo",
        "UserId": "nvbot-evaluation",
        "Env": "stg",
        "PlatformConfig": {
            "ConfigType": "api",
            "URL": "https://talktobugs-stg.nvidia.com/talk_to_your_bugs/query_eval/",
            "Payload": {
                "flags": {
                    "enable_debug_info": true,
                    "hybrid_search_fallback_in_agent": false,
                    "enable_hybrid_fallback": false
                }
            }
        }
    }
    ```

    10. auto_help
    ```
    {
        "Project": "auto_help_evaluation",
        "RunType": "manual",
        "System": "auto_help",
        "Model": "mixtral",
        "UserId": "nvbot-evaluation",
        "Env": "stg",
        "PlatformConfig": {
            "ConfigType": "api",
            "URL": "https://nvidiadev.service-now.com/api/gnvi2/custom_skills/auto_help_clone",
            "Auth": {
                "AuthType": "basic",
                "Username": "skilladmin",
                "Password": ""
            },
            "Payload": {"short_description": ""}
        }
    }
    ```

    11. finance
    ```
    {
        "Project": "finance_ai_earnings_v2_mixtral_complete_evaluation",
        "RunType": "manual",
        "System": "finance_ai_earnings_v2",
        "Model": "mixtral",
        "Attachments": [],
        "UserId": "nvbot-evaluation",
        "Env": "sandbox",
        "Parameters": {
                "tag1": ""
        }
    }
    ```

    """

    # if PlatformConfig is specified, query directly to /chatbot
    # else fetch config from config manager

    project = run_request.Project
    if project:
        logger.info(f"Started run for ProjectId: {run_request.Project}")
    span = tracer.current_span()
    logger.info(f"span: {span}")
    trace_id = str(span.trace_id) if span else "None"
    logger.info(f"trace_id: {trace_id}")

    print(f"Run request: {run_request.dict()}")
    logger.info(f"Run request: {run_request.dict()}")
    env = run_request.Env
    try:
        ### --- Fetch configs --- ###
        # TODO: fetch platform if PlatformConfig not provided in run_request
        platform_config = run_request.PlatformConfig
        project_id = run_request.ProjectId
        evaluation_project = get_evaluation_project_by_id(run_request.ProjectId)
        if project:
            logger.info(f"Started run for ProjectId: {run_request.ProjectId}")

        if run_request.PlatformConfig is None:
            # try fetch evaluation project first, if failed, fetch bot config
            json_platform_config = await get_platform_config(run_request, evaluation_project)
            assert json_platform_config is not None, f"Failed to fetch bot config for {run_request.Project}, with model {run_request.Model} and system {run_request.System}."
            run_request.PlatformConfig = BotPlatformConfig.model_validate(json_platform_config)
            project = project if project is not None else evaluation_project['project_name']

        if run_request.EvaluationConfig is None:
            json_eval_config = await get_evaluation_config(run_request, evaluation_project)
            assert json_eval_config is not None, f"Failed to fetch eval config for {run_request.Project}, with model {run_request.Model} and system {run_request.System}."
            run_request.EvaluationConfig = NVBotEvaluationConfig.model_validate(json_eval_config)
            # assert json_eval_config is not None, f"Failed to fetch eval config for {run_request.ProjectId}, with {run_request.Project} asset."
            # run_request.EvaluationConfig = NVBotEvaluationConfig.model_validate(json_eval_config)

        eval_config = run_request.EvaluationConfig

        if eval_config and run_request.Customization:
            # {Notification: {EmailRecipients: []}
            updated_schema = update_dict(eval_config.model_dump(exclude_unset=True), run_request.Customization, overwrite=True)
            eval_config = NVBotEvaluationConfig.model_validate(updated_schema)
            run_request.EvaluationConfig = eval_config

        if eval_config:
            regression_schema = eval_config.RegressionSchema
            assert regression_schema is not None, f"Failed to find custom eval due to lack of regression schema for {run_request.ProjectId}"

            ### --- Prepare Answer Generation --- ###
            # print("⭐Start Regression run ")
            rc = RegressionRunContainer(
                project=project,
                config=eval_config,
                env=env
            )
            db_response = await rc.initiate(request=run_request)
            print("Initiate response", db_response)
            return JSONResponse(status_code=201, content=jsonable_encoder(db_response, by_alias=False))

    except Exception as ex:
        logger.error(f"Error when initiating nemo regression run, {ex}")
        return JSONResponse(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                            content=jsonable_encoder(f"Error: {ex}"))


@router.post("/start_run")
async def start_run():
    span = tracer.current_span()
    logger.info(f"span: {span}")
    print(f"start_run span: {span}")
    trace_id = str(span.trace_id) if span else "None"
    logger.info(f"start_run trace_id: {trace_id}")
    print(f"start_run trace_id: {trace_id}")

    # with tracer.trace("router.request") as span:
    #     trace_id = span.trace_id  # Get the trace ID
    #     logger.info(f"start_run router.request  trace_id: {trace_id}")
    #     print(f"start_run router.request trace_id: {trace_id}")

    filters = _convert_filters_format({"status": EvaluationRunStatus.IN_PROCESS.value})
    logger.info(f"Filters: {filters}")
    print("filters", filters)
    evaluation_history_list = DatabaseHandler({'env': 'dev'}).get_evaluation_history_records(filters).get("items")
    if evaluation_history_list is None:
        return JSONResponse(status_code=500, content=jsonable_encoder("Failed to connect to Database", by_alias=False))
    # on complete, update status
    results = {}
    logger.info(f"⌛️Received {len(evaluation_history_list)} evaluation history list ")
    print(f"⌛️Received {len(evaluation_history_list)} evaluation history list ")
    for history in evaluation_history_list:
        history_id = history.get("id")
        # filter datetime

        span = tracer.current_span()
        logger.info(f"history span: {span}")
        print(f"history span : {span}")
        trace_id = str(span.trace_id) if span else "None"
        logger.info(f"history trace_id: {trace_id}")
        print(f"history trace_id : {trace_id}")

        evaluation_history_metadata = json.loads(history.get("metadata_value"))
        env = FlowConfigRequest.model_validate(evaluation_history_metadata).Env

        project = str(history.get("project", ""))
        project_id = history.get("project_id", None)
        try:
            eval_config = NVBotEvaluationConfig.model_validate(history.get("evaluation_metadatajson", {}))
            platform_config = get_platform_config_model_from_dict(history.get("flowconfig_metadatajson", {}))
            evaluation_history_metadata = json.loads(history.get("metadata_value"))
            env = evaluation_history_metadata.get("Env")
            run_request = RunMakerRequest(
                Attachments=[],
                Project=project,
                ProjectId=history.get("project_id", None),
                RunType=history.get("run_type"),
                UserId=evaluation_history_metadata.get("UserId", ""),
                UserName=evaluation_history_metadata.get("UserName", ""),
                Parameters={"tag1": history.get("tag1"), "tag2": history.get("tag2")},
                EvaluationConfig=eval_config,
                System=FlowConfigRequest.model_validate(evaluation_history_metadata).System,
                Model=FlowConfigRequest.model_validate(evaluation_history_metadata).Model,
                ConfigId=FlowConfigRequest.model_validate(evaluation_history_metadata).ConfigId,
                Env=FlowConfigRequest.model_validate(evaluation_history_metadata).Env
            )
            run_request.PlatformConfig = platform_config
            print(f"⭐Start Regression run {project}")

            try:
                history_id = history.get("id")
                param = UserEvaluationRunData(
                    Id=history_id,
                    Project=project,
                    NtAccount=EVAL_ACCOUNT_ID,
                    Username=NT_ACCOUNT_NAME,
                    Status=EvaluationRunStatus.STARTED.value,
                    OutputUrl=history.get("output_url", ""),
                    RunType=history.get("run_type", ""),
                    EvalType=history.get("eval_type", ""),
                    Tag1=history.get("tag1", ""),
                    Tag2=history.get("tag2", ""),
                    Trace=history.get("trace", ""),
                )
                update_response = DatabaseHandler({'env': 'dev'}).update_evaluation_history_and_details(param)
                assert update_response, f"Failed to update database for history and details info: {update_response}"
            except Exception as ex:
                logger.error(f"Failed to update database for history and details info: {ex}")
                continue
            rc = RegressionRunContainer(
                project=project,
                config=eval_config,
                env=env
            )
            dataframe = await rc.prepare(request=run_request, config=eval_config)
            regression_run_results = await rc.arun(request=run_request, config=eval_config, df=dataframe)
            regression_result_file = regression_run_results.get("local_filepath")

            print("Local file ", regression_result_file)

            if not eval_config.EvaluationSchema:
                return JSONResponse(status_code=200, content=jsonable_encoder(regression_run_results, by_alias=False))
            else:
                if DatasetConfig not in eval_config.EvaluationSchema.NemoEvaluator:
                    eval_config.EvaluationSchema.NemoEvaluator.DatasetConfig = DatasetConfig()
                eval_config.EvaluationSchema.NemoEvaluator.DatasetConfig.Engine = StorageType.LOCAL.value
                eval_config.EvaluationSchema.NemoEvaluator.DatasetConfig.RunFile = regression_result_file
        except Exception as ex:
            logger.error(f"Error when running nemo regression evaluation, {ex}")

            try:
                param = UserEvaluationRunData(
                    Id=history_id,
                    Project=project,
                    NtAccount=EVAL_ACCOUNT_ID,
                    Username=NT_ACCOUNT_NAME,
                    Status=EvaluationRunStatus.FAILED.value
                )
                update_response = DatabaseHandler({'env': 'dev'}).update_evaluation_history_and_details(param)
                assert update_response, f"Error returned to update database for history and details info: {update_response}"

            except Exception as ex:
                logger.error(f"Failed to update database for history and details info: {ex}")

            return JSONResponse(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                                content=jsonable_encoder(f"Error: {ex}"))

        try:
            ### --- Start Nemo Evaluator Evaluation --- ###
            print(f"⭐Start Nemo Evaluator Evaluation {project}")
            logger.info("⭐Start Nemo Evaluator Evaluation")

            nec = NemoEvaluationRunContainer(
                project=project,
                config=eval_config,
                env=env,
            )
            # remove prev history
            await nec.initiate(request=run_request, evaluation_history=history)
            nec.prepare(request=run_request, config=eval_config)
            nec_results = await nec.arun(request=run_request, config=eval_config)
        except Exception as ex:
            logger.error(f"Error when running nemo evaluator evaluation, {ex}")
            return JSONResponse(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                                content=jsonable_encoder(f"Error: {ex}"))

        # results = nec.finish(request=run_request, config=eval_config)
        # eval_results = nec.finish(request=run_request, config=eval_config, results = results)
        results[history.get("id")] = nec_results

        run_type = history.get("run_type")
        project = history.get("project")
        project_id = history.get("project_id")
        if run_type.startswith("cron"):
            filters = {"run_type": run_type, "project": project, "limit": 2}
            previous_runs = DatabaseHandler({'env': 'dev'}).get_evaluation_history_records(
                filters=filters,
                pagination={
                    "page": 1,
                    "size": 2
                }
            ).get("items")

            # form datasetconfigs

    return JSONResponse(status_code=200, content=jsonable_encoder(results, by_alias=False))


async def run_custom_eval(history: dict, json_eval_config: dict, custom_eval_config: NVBotEvaluationConfig):
    nemo_eval_id = json.loads(history.get("output_url", {})).get("id", "")
    try:
        ### --- Start Nemo Evaluator Evaluation --- ###
        print("⭐ Run custom evaluation")
        logger.info("Start Nemo Evaluator Evaluation")
        metadata_json = history.get("metadata_value")
        run_request = RunMakerRequest.model_validate(json.loads(metadata_json))
        run_request.EvaluationConfig = NVBotEvaluationConfig.model_validate(custom_eval_config)
        run_request.PlatformConfig = BotPlatformConfig.model_validate(history.get("flowconfig_metadatajson", {}))
        project = run_request.Project

        project = str(history.get("project", ""))

        evaluation_history_metadata = json.loads(history.get("metadata_value"))
        env = FlowConfigRequest.model_validate(evaluation_history_metadata).Env

        nec = NemoEvaluationRunContainer(
            project=project,
            config=custom_eval_config,
            env=env
        )
        response = await nec.post_processing(history, nemo_eval_id)
        print(f"post_processing_evaluation  response: {response}")
        result_dataset_config = response.get("result_dataset_config")
        print(f"custom dataset config: {result_dataset_config}")
        assert result_dataset_config is not None, f"Failed to get local directory for {LLM_AS_A_JUDGE_EVALUATOR_TYPE} from result_dataset_config"
        result_dataset_config = DatasetConfig.model_validate(result_dataset_config)

        custom_eval_config.EvaluationSchema.NemoEvaluator.DatasetConfig = DatasetConfig.model_validate({
            "Engine": "local",
            "RunFile": result_dataset_config.RunFile
        })

        nec.prepare(request=run_request, config=custom_eval_config)
        results = await nec.arun(request=run_request, config=custom_eval_config)

        return results
    except Exception as ex:
        logger.error(f"Error when running nemo regression evaluation, {ex}")
        return JSONResponse(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                            content=jsonable_encoder(f"Error: {ex}"))


@router.post("/regression_run")
async def regression_run(run_request: RunMakerRequest, use_nemo_inference: bool = False):
    """
    Run regression evaluation. \n
    Sample request: \n
    1.
    ```
    {
        "Project": "nvbot_for_nvhelp_mixtral_agent_complete_evaluation",
        "RunType": "manual",
        "System": "nvhelp",
        "Model": "mixtral_agent",
        "Attachments": [],
        "Env": "dev"
    }
    ```
    \n
    2.
    ```
    {
        "Project": "avc_mixtral_sample_custom_evaluation",
        "RunType": "manual",
        "System": "avc",
        "Model": "mixtral",
        "Attachments": [],
        "Env": "dev"
    }
    ```
    \n
    3.
    ```
    {
        "Project": "nvinfo_llama4_complete_evaluation",
        "RunType": "manual",
        "System": "nvinfo",
        "Model": "llama4",
        "UserId": "nvbot-evaluation",
        "Env": "dev",
    }
    ```
    \n
    4.
    ```
    {
        "Project": "scout_mixtral_complete_evaluation",
        "RunType": "manual",
        "System": "scout_long",
        "Model": "mixtral",
        "Attachments": [],
        "UserId": "nvbot-evaluation",
        "Env": "sandbox"
    }
    ```
    \n
    5.
    ```
    {
        "Project": "orchestrator_perceptor_complete_evaluation",
        "RunType": "manual",
        "System": "orchestrator_perceptor",
        "Model": "llama_3_1",
        "Attachments": [],
        "UserId": "nvbot-evaluation",
        "Env": "sandbox"
    }
    ```
    \n
    6. Local runner for quick testing
    ```
    {
        "Project": "nvinfo_holiday_expert_complete_evaluation",
        "RunType": "manual",
        "System": "nvinfo_holiday_expert",
        "Model": "llama3_agent",
        "UserId": "yangj",
        "Env": "sandbox",
        "Parameters": {"tag1": "GPT4"}
    }
    ```
    \n
    
    7.scout_long
    ```
    {
        "Project": "scout_long_mixtral_complete_evaluation",
        "RunType": "manual",
        "System": "scout_long",
        "Model": "mixtral",
        "Attachments": [],
        "UserId": "nvbot-evaluation",
        "Env": "stg"
    }
    ```

    \n

    8.global protect
    ```
    {
        "Project": "gp_complete_evaluation",
        "RunType": "manual",
        "System": "globalprotect",
        "Model": "mixtral",
        "UserId": "nvbot-evaluation",
        "Env": "stg"
    }
    ```

    9. va
    ```
    {
        "Project": "va_complete_evaluation",
        "RunType": "manual",
        "System": "va",
        "Model": "mixtral",
        "UserId": "nvbot-evaluation",
        "Env": "stg",
        "PlatformConfig": {
            "ConfigType": "api",
            "URL": "https://nvidiastage.service-now.com/api/now/vaconversationalapis/getResponse",
            "Auth": {
                "AuthType": "basic",
                "Username": "testing.user",
                "Password": "Password.123"
            },
            "Payload": {}
        }
    }
    ```

    10. nvbugs
    ```
    {
        "Project": "nvbugs_complete_evaluation",
        "RunType": "manual",
        "System": "bugnemo",
        "UserId": "nvbot-evaluation",
        "Env": "sandbox",
        "PlatformConfig": {
            "ConfigType": "api",
            "URL": "http://127.0.0.1:10000/talk_to_your_bugs/query_eval/",
            "Payload": {
                "flags": {
                    "enable_debug_info": true
                }
            }
        }
    }
    ```

    11. astra_assist

    ```
    {
            "Project": "astra_assist_evaluation",
            "RunType": "manual",
            "UserId": "nvbot-evaluation",
            "System": "astra",
            "Env": "dev",
            "PlatformConfig": {
                "ConfigType": "api",
                "URL": " https://https://natsep15.stg.astra.nvidia.com/generate",
                "Auth": {
                    "AuthType": "starfleet",
                    "Env": "prd"
                },
                "Payload": {}
            },
            "Parameters": {}
        }
    ```
    """

    # if PlatformConfig is specified, query directly to /chatbot
    # else fetch config from config manager

    span = tracer.current_span()
    logger.info(f"span: {span}")
    print(f"regression_run span: {span}")
    trace_id = str(span.trace_id) if span else "None"
    logger.info(f"regression_run trace_id: {trace_id}")
    print(f"regression_run trace_id: {trace_id}")

    context["trace_id"] = str(trace_id)

    project = run_request.Project
    if project:
        logger.info(f"Started regression run for Project: {run_request.Project}")

    print(f"Run request: {run_request.dict()}")
    logger.info(f"Run request: {run_request.dict()}")
    env = run_request.Env
    json_eval_config = None
    nec = None

    try:
        ### --- Fetch configs --- ###
        # TODO: fetch platform if PlatformConfig not provided in run_request
        platform_config = run_request.PlatformConfig
        project_id = run_request.ProjectId
        evaluation_project = get_evaluation_project_by_id(run_request.ProjectId)
        if project_id:
            logger.info(f"Started regression run for ProjectId: {run_request.ProjectId}")
        # fetch configs if PlatformConfig not provided in run_request
        # if run_request.PlatformConfig is None:
        #     # try fetch evaluation project first, if failed, fetch bot config
        #     json_platform_config = await get_platform_config(run_request, evaluation_project)
        #     assert json_platform_config is not None, f"Failed to fetch bot config for {run_request.Project}, with model {run_request.Model} and system {run_request.System}."
        #     run_request.PlatformConfig = NVBotPlatformConfig.model_validate(json_platform_config)
        #
        # # assert platform_config is not None, f"Failed to fetch bot config for {run_request.Project}, with model {run_request.Model} and system {run_request.System}."
        # # run_request.PlatformConfig = NVBotPlatformConfig.model_validate(platform_config)
        #
        # print(f"Fetched flow Config file for evaluation project id {project_id}")
        # # TODO: migrate to config manager repo
        # json_eval_config = None
        # if run_request.EvaluationConfig is None:
        #     if evaluation_project and evaluation_project.get("asset_metadatajson", {}) != {}:
        #         json_eval_config = evaluation_project.get("asset_metadatajson")
        #     else:
        #         with open(f'asset/{project}.json', 'r') as json_file:
        #             json_eval_config = json.load(json_file)
        if run_request.PlatformConfig is None:
            # try fetch evaluation project first, if failed, fetch bot config
            # ONLY available for platform dependent bots
            json_platform_config = await get_platform_config(run_request, evaluation_project)
            assert json_platform_config is not None, f"Failed to fetch bot config for {run_request.Project}, with model {run_request.Model} and system {run_request.System}."
            run_request.PlatformConfig = BotPlatformConfig.model_validate(json_platform_config)

        if run_request.EvaluationConfig is None:
            json_eval_config = await get_evaluation_config(run_request, evaluation_project)
            assert json_eval_config is not None, f"Failed to fetch eval config for {run_request.Project}, with model {run_request.Model} and system {run_request.System}."
            run_request.EvaluationConfig = NVBotEvaluationConfig.model_validate(json_eval_config)

        assert run_request.EvaluationConfig is not None, f"Failed to fetch eval config for {run_request.ProjectId}, with {run_request.Project} asset."

        eval_config = run_request.EvaluationConfig
        if eval_config and run_request.Customization:
            # {Notification: {EmailRecipients: []}
            updated_schema = update_dict(eval_config.model_dump(exclude_unset=True), run_request.Customization, overwrite=True)
            eval_config = NVBotEvaluationConfig.model_validate(updated_schema)
            run_request.EvaluationConfig = eval_config

        if use_nemo_inference == False:
            regression_schema = eval_config.RegressionSchema
            assert regression_schema is not None, f"Failed to find custom eval due to lack of regression schema for {run_request.ProjectId}"

            ### --- Prepare Answer Generation --- ###

            print(f"⭐Start Regression run {project}")
            rc = RegressionRunContainer(
                project=project,
                config=eval_config,
                env=env
            )

            dataframe = await rc.prepare(request=run_request, config=eval_config)
            run_results = await rc.arun(request=run_request, config=eval_config, df=dataframe)
            regression_result_file = run_results.get("local_filepath")

            print("Local file ", regression_result_file)

            if not eval_config.EvaluationSchema or not eval_config.EvaluationSchema.NemoEvaluator:
                return JSONResponse(status_code=200, content=jsonable_encoder(run_results, by_alias=False))
            else:
                if DatasetConfig not in eval_config.EvaluationSchema.NemoEvaluator:
                    eval_config.EvaluationSchema.NemoEvaluator.DatasetConfig = DatasetConfig()
                eval_config.EvaluationSchema.NemoEvaluator.DatasetConfig.Engine = StorageType.LOCAL.value
                eval_config.EvaluationSchema.NemoEvaluator.DatasetConfig.RunFile = regression_result_file

        ### --- Start Nemo Evaluator Evaluation --- ###
        print(f"⭐Start Nemo Evaluator Evaluation {project}")
        logger.info("⭐Start Nemo Evaluator Evaluation")

        nec = NemoEvaluationRunContainer(
            project=project,
            config=eval_config,
            env=env,
        )
        nec.prepare(request=run_request, config=eval_config)
        results = await nec.arun(request=run_request, config=eval_config)
        # results = nec.finish(request=run_request, config=eval_config)
        # eval_results = nec.finish(request=run_request, config=eval_config, results = results)

        print("Eval run results", results)

    except Exception as ex:
        logger.error(f"Error when running nemo regression evaluation, {ex}")
        if rc:
            rc.finish(request=run_request, config=eval_config, run_results={"status": "exception", "error": f"{ex}"})
        return JSONResponse(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                            content=jsonable_encoder(f"Error: {ex}"))

    return JSONResponse(status_code=200, content=jsonable_encoder(results, by_alias=False))


@router.post("/eval_run")
async def eval_run(run_request: RunMakerRequest):
    project = run_request.Project
    if project:
        logger.info(f"Started eval run for Project: {run_request.Project}")
    span = tracer.current_span()
    logger.info(f"tracer id: {span.trace_id if span else 'None'}")
    print(f"Run request: {run_request.dict()}")
    logger.info(f"Run request: {run_request.dict()}")
    env = run_request.Env
    eval_config = run_request.EvaluationConfig
    nec = None

    try:
        ### --- Fetch configs --- ###
        # TODO: fetch platform if PlatformConfig not provided in run_request
        platform_config = run_request.PlatformConfig
        project_id = run_request.ProjectId
        evaluation_project = get_evaluation_project_by_id(run_request.ProjectId)
        if project_id:
            logger.info(f"Started regression run for ProjectId: {run_request.ProjectId}")

        print(f"⭐Start Nemo Evaluator Evaluation {project}")
        logger.info("⭐Start Nemo Evaluator Evaluation")

        json_eval_config = None
        if run_request.EvaluationConfig is None:
            json_eval_config = await get_evaluation_config(run_request, evaluation_project)
            assert json_eval_config is not None, f"Failed to fetch eval config for {run_request.Project}, with model {run_request.Model} and system {run_request.System}."
            run_request.EvaluationConfig = NVBotEvaluationConfig.model_validate(json_eval_config)


        eval_config = run_request.EvaluationConfig
        if eval_config and run_request.Customization:
            # {Notification: {EmailRecipients: []}
            updated_schema = update_dict(eval_config.model_dump(exclude_unset=True), run_request.Customization, overwrite=True)
            eval_config = NVBotEvaluationConfig.model_validate(updated_schema)
            run_request.EvaluationConfig = eval_config

        # Hack: if want to run from local
        # eval_config.EvaluationSchema.NemoEvaluator.DatasetConfig.Engine = StorageType.LOCAL.value
        # eval_config.EvaluationSchema.NemoEvaluator.DatasetConfig.RunFile = "Glean_compare_may7_2025.xlsx"

        nec = NemoEvaluationRunContainer(
            project=project,
            config=eval_config,
            env=env,
        )
        nec.prepare(request=run_request, config=eval_config)
        results = await nec.arun(request=run_request, config=eval_config)

        print("Eval run results", results)

    except Exception as ex:
        logger.error(f"Error when running nemo regression evaluation, {ex}")
        if nec:
            nec.finish(request=run_request, config=eval_config, run_results={"status": "exception", "error": f"{ex}"})
        return JSONResponse(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                            content=jsonable_encoder(f"Error: {ex}"))

    return JSONResponse(status_code=200, content=jsonable_encoder(results, by_alias=False))


# async def regression_autorun_after_eval_run(process_evaluation_request: EvaluationProcessingRequest):
#     """
#         this will run the evaluation list which satisfies the requirement in request parameter
#         once llm-as-a-judge completes, checks on evaluation schema, and runs custom evaluation if evalutor is defined.
#
#     Example: \n
#     ```
#     {
#         "CreatedDateFrom": "-1",
#         "Status": "STARTED"
#
#     }
#     ```
#     """
#     filters = _convert_filters_format(process_evaluation_request.dict())
#     logger.info(f"Filters: {filters}")
#     print("filters", filters)
#
#     evaluation_history_list = DatabaseHandler({'env': 'dev'}).get_evaluation_history_records(filters).get("items")
#     if evaluation_history_list is None:
#         return JSONResponse(status_code=500, content=jsonable_encoder("Failed to connect to Database", by_alias=False))
#     # on complete, update status
#     results = {}
#     logger.info(f"👌Received {len(evaluation_history_list)} evaluation history list ")
#     print(f"👌Received {len(evaluation_history_list)} evaluation history list ")
#     for history in evaluation_history_list:
#         history_id = history.get("id")
#
#         if history_id is not None:
#             evaluation_history_metadata = json.loads(history.get("metadata_value"))
#             env = FlowConfigRequest.model_validate(evaluation_history_metadata).Env
#
#             eval_metadata = history.get("evaluation_metadatajson", {})
#             eval_config = NVBotEvaluationConfig.model_validate(eval_metadata)
#             nec = NemoEvaluationRunContainer(
#                 project=history.get('project', ""),
#                 config=eval_config,
#                 env=env,
#             )
#
#             nemo_eval_id = json.loads(history.get("output_url", {})).get("id", "")
#             nemo_eval_type = json.loads(history.get("output_url", {})).get("eval_type", "")
#
#             if not nemo_eval_id:
#                 logger.info(f"Failed getting nemo_eval_id {history_id}")
#                 continue
#
#             if history.get("status") in [EvaluationRunStatus.FAILED.value]:
#                 continue
#
#             if nemo_eval_type.lower() == LLM_AS_A_JUDGE_EVALUATOR_TYPE:
#                 json_eval_config = history.get("evaluation_metadatajson", {})
#                 updated_eval_config = copy.deepcopy(NVBotEvaluationConfig.model_validate(json_eval_config))
#                 evaluator = get_evaluation_config_by_nemo_evaluator_type(eval_config.EvaluationSchema,
#                                                                          CUSTOM_EVALUATOR_TYPE)
#                 # check if custom evaluator exist
#                 if evaluator is not None:
#                     updated_eval_config.EvaluationSchema.NemoEvaluator.Evaluators = [evaluator]
#
#                     results[nemo_eval_id] = await run_custom_eval(history, json_eval_config, updated_eval_config)
#                 else:
#                     logger.info(f"Post processing evaluation result {nemo_eval_id} for {nemo_eval_type}")
#                     response = await nec.post_processing_evaluation(history, nemo_eval_id)
#                     results[nemo_eval_id] = response
#
#     return JSONResponse(status_code=200, content=jsonable_encoder(results, by_alias=False))


@router.post("/comparison_run")
async def comparison_run(run_request: RunMakerRequest):
    """
    Sample request: \n
    1.
    ```
    {
      "Project": "nvinfo_mixtral_agent_consistency_evaluation",
      "UserId": "nvbot-evaluation",
      "Customization": {
        "DatasetConfigs": [
          {
            "Engine": "database",
            "HistoryId": "3369"
          },
          {
            "Engine": "database",
            "HistoryId": "3371"
          },
          {
            "Engine": "database",
            "HistoryId": "3415"
          }
        ]
      }
    }
    ```
    \n
    """
    json_eval_config = None
    if run_request.EvaluationConfig is None:
        json_eval_config = await ConfigLoader().get_evaluation_schema(run_request.Project)
        if json_eval_config:
            print("get eval config from ConfigLoader")
        else:
            return None
    eval_config = NVBotEvaluationConfig.model_validate(json_eval_config)

    if not is_valid_instance(eval_config.ComparisonSchema):
        created_column_map = {"scorers": []}
        for evaluator in eval_config.EvaluationSchema.NemoEvaluator.Evaluators:
            created_column_map.update(evaluator.get("column_map", {}))
            if "judge_config" in evaluator:
                created_column_map["scorers"].extend(evaluator.get("judge_config").get("scorers", []))

        eval_config.ComparisonSchema = ComparisonSchema(
            DataConfigs={"column_map": created_column_map}
        )

    if is_valid_instance(eval_config) and run_request.Customization:
        # {Notification: {EmailRecipients: []}
        # {DatasetConfigs: [{DatasetId: "", ]}
        comparison_schema = update_dict(eval_config.ComparisonSchema.model_dump(exclude_unset=True), run_request.Customization)
        eval_config.ComparisonSchema = ComparisonSchema.model_validate(comparison_schema)


    run_request.EvaluationConfig = eval_config
    assert len(
        run_request.EvaluationConfig.ComparisonSchema.DatasetConfigs) > 0, "Expect dataset configs configuration explicitly for comparision."

    cc = ComparisonContainer(
        project=run_request.Project,
        config=eval_config,
    )

    initiate_response = cc.initiate(run_request)
    history_id = initiate_response.get("id", "")

    try:
        # overwrite
        # json_eval_config = cc.config
        prepared_df = await cc.prepare(request=run_request, config=eval_config)
        run_results = await cc.arun(request=run_request, config=eval_config, df=prepared_df)
        results = cc.finish(request=run_request, config=eval_config, run_results=run_results,
                            evaluation_history=initiate_response)

    except Exception as ex:
        try:
            param = UserEvaluationRunData(
                Id=history_id,
                Project=run_request.project,
                NtAccount=EVAL_ACCOUNT_ID,
                Username=NT_ACCOUNT_NAME,
                Status=EvaluationRunStatus.FAILED.value
            )
            update_response = DatabaseHandler({'env': 'dev'}).update_evaluation_history_and_details(param)
            assert update_response, f"Error returned on update database for history and details info: {update_response}"

        except Exception as ex:
            logger.error(f"Failed to update database for history and details info: {ex}")

    return JSONResponse(status_code=200, content=jsonable_encoder(results, by_alias=False))

@router.post("/variability_run")
async def variability_run(run_request: RunMakerRequest):

    customization = run_request.Customization
    repeat = customization.get("repeat")

    pass

@router.post("/post_processing")
async def eval_post_processing(process_evaluation_request: EvaluationProcessingRequest):
    """
    Post fetch evaluation results, persist nemo evaluation output in database, and send notification. \n
    Sample request: \n
    1. {
        "RunType": "cron"
    }
    \n
    2. {
        "Project": "nvbot_for_nvhelp_mixtral_agent_sample",
        "CreatedDateFrom": "{\\\"days\\\": -3}",
        "Status": "STARTED"
    }

    \n
    """

    filters = _convert_filters_format(process_evaluation_request.dict())
    logger.info(f"Filters: {filters}")
    print("filters", filters)
    evaluation_history_list = DatabaseHandler({'env': 'dev'}).get_evaluation_history_records(filters).get("items")
    if evaluation_history_list is None:
        return JSONResponse(status_code=500, content=jsonable_encoder("Failed to connect to Database", by_alias=False))
    # on complete, update status
    results = {}
    logger.info(f"👌Received {len(evaluation_history_list)} evaluation history list ")
    print(f"👌Received {len(evaluation_history_list)} evaluation history list ")
    for history in evaluation_history_list:
        history_id = history.get("id")
        # filter datetime

        evaluation_history_metadata = json.loads(history.get("metadata_value"))
        env = FlowConfigRequest.model_validate(evaluation_history_metadata).Env
        if history_id is not None:
            eval_metadata = history.get("evaluation_metadatajson", {})
            eval_config = NVBotEvaluationConfig.model_validate(eval_metadata)

            nec = NemoEvaluationRunContainer(
                project=history.get('project', ""),
                config=eval_config,
                env=env
            )
            print(history.get("output_url", {}))
            output_url_string = history.get("output_url", {})
            if not output_url_string:
                continue
            nemo_eval_id = json.loads(output_url_string).get("id", "")
            if not nemo_eval_id:
                logger.info(f"Failed getting nemo_eval_id {history_id}")
                continue

            nemo_eval_type = json.loads(history.get("output_url", {})).get("eval_type", "")

            if history['status'] in [EvaluationRunStatus.COMPLETED.value]:
                continue

            logger.info(f"Post processing evaluation result {nemo_eval_id} for {nemo_eval_type}")
            response = await nec.post_processing(history, nemo_eval_id)

            results[nemo_eval_id] = response
            # break
    return JSONResponse(status_code=200, content=jsonable_encoder(results, by_alias=False))
