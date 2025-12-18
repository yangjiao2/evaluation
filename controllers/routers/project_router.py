import http
import shutil

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

import base64
from typing import List, Optional, Dict
from pydantic import BaseModel, validator

from data_models.api.run_maker import RunMakerRequest, FlowConfigRequest, DatasetConfig, ColumnMapModel, JudgeModel, \
    StorageType

import json
import logging
import os
import uuid
import time
import requests

from configs.settings import get_settings
from nvbot_models.request_models.evaluation_request import UserEvaluationProject, UserEvaluationDataset
from nvbot_utilities.utils import api_handler
from service_library.constants import NT_ACCOUNT_ID, LOCAL_TMP_FOLDER, REFERENCE_MODEL, LLM_AS_A_JUDGE_EVALUATOR_TYPE, \
    DEFAULT_JUDGE_PARAM, DATASET_FILENAME
from service_library.handler.config_loader import ConfigLoader
from service_library.handler.database_handler import DatabaseHandler
from service_library.handler.s3_dataset_handler import S3DatasetHandler
from service_library.nemo_ms.nemo_datastore import NeMoDataStore
from service_library.nemo_ms.nemo_service_helper import prepare_llm_as_a_judge_nemoeval_dataset
from service_library.run.parser_library.input_parser import prepare_chatbot_request, generate_fulfillment_request

from service_library.run.regression_container import RegressionRunContainer
from service_library.utils.configuration_helper import get_evaluation_project_by_id
from service_library.utils.data_helper import move_file
from service_library.utils.request_helpers import create_header
from service_library.utils.run_helpers import generate_unique_datastore_path, convert_to_snake_case

router = APIRouter(
    tags=["Project"]
)

log = logging.getLogger('Evaluation Project Router')

PAYLOAD = {
    "PlatformBasedConfig": {
        # "nvhelp": {
        #     "Project": "nvbot_for_nvhelp_mixtral_agent_complete_evaluation",
        #     "RunType": "manual",
        #     "System": "nvhelp",
        #     "Model": "mixtral_agent",
        #     "UserId": "nvbot-evaluation",
        #     "Env": "dev",
        #     "Parameters": {}
        # },
        "nvinfo_llama4": {
            "Project": "nvinfo_llama4_complete_evaluation",
            "RunType": "manual",
            "System": "nvinfo",
            "Model": "llama4",
            "UserId": "nvbot-evaluation",
            "Env": "dev",
            "Parameters": {}
        },
        "scout_long": {
            "Project": "scout_long_mixtral_complete_evaluation",
            "RunType": "manual",
            "System": "scout_long",
            "Model": "mixtral",
            "UserId": "nvbot-evaluation",
            "Env": "sandbox",
            "Parameters": {}
        },
        "perceptor": {
            "Project": "orchestrator_perceptor_complete_evaluation",
            "RunType": "manual",
            "System": "orchestrator_perceptor",
            "Model": "llama_3_1",
            "UserId": "nvbot-evaluation",
            "Env": "sandbox",
            "Parameters": {}
        },
        "developer_knowledge_expert": {
            "Project": "developer_knowledge_expert_complete_evaluation",
            "RunType": "manual",
            "System": "developer_knowledge_expert",
            "Model": "llama_3_1_agent_graph",
            "UserId": "nvbot-evaluation",
            "Env": "dev",
            "Parameters": {}
        }
    },
    "APIBasedConfig": {
        "bugnemo": {
            "Project": "nvbugs_complete_evaluation",
            "RunType": "manual",
            "UserId": "nvbot-evaluation",
            "System": "bugnemo",
            "Env": "stg",
            "PlatformConfig": {
                "ConfigType": "api",
                "URL": "https://talktobugs-stg.nvidia.com/talk_to_your_bugs/query_eval/",
                "Payload": {
                    "flags": {
                        "enable_debug_info": True,
                        "hybrid_search_fallback_in_agent": False,
                        "enable_hybrid_fallback": False,
                        "is_evaluator_pipeline": True
                    }
                }
            },
            "Parameters": {},
        },
        "sn_virtualagent": {
            "Project": "va_complete_evaluation",
            "RunType": "manual",
            "System": "sn_virtualagent",
            "UserId": "nvbot-evaluation",
            "Env": "stg",
            "PlatformConfig": {
                "ConfigType": "api",
                "URL": "https://nvidiastage.service-now.com/api/now/vaconversationalapis/getResponse",
                "Auth": {
                    "AuthType": "basic",
                    "Username": "testing.user",
                    "Password": ""
                },
                "Payload": {}
            }
        },
        "auto_help": {
            "Project": "auto_help_evaluation",
            "RunType": "manual",
            "System": "auto_help",
            "Model": "mixtral",
            "UserId": "nvbot-evaluation",
            "Env": "dev",
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
        },
        "it_support_agent": {
            "Project": "it_support_agent_evaluation",
            "RunType": "manual",
            "UserId": "nvbot-evaluation",
            "System": "it_support_agent",
            "Env": "stg",
            "PlatformConfig": {
                "ConfigType": "api",
                "URL": "https://itsupportagent-stg.nvidia.com/generate",
                "Auth": {
                    "AuthType": "custom",

                    "Payload": {
                        "x-auth-mode": "nvauth",
                        "x-nvauth-system-account-token": "$IT_SUPPORT_NVAUTH_ACCOUNT_TOKEN$",
                        "Authorization": "Bearer $IT_SUPPORT_BEARER_TOKEN$"
                    }
                },
                "Payload": {}
            },
            "Parameters": {}
        },
        "astra_assist": {
            "Project": "astra_assist_evaluation",
            "RunType": "manual",
            "UserId": "nvbot-evaluation",
            "System": "astra",
            "Env": "dev",
            "PlatformConfig": {
                "ConfigType": "api",
                "URL": " https://natsep15.stg.astra.nvidia.com/generate",
                "Auth": {
                    "AuthType": "starfleet",
                    "Env": "prd"
                },
                "Payload": {}
            },
            "Parameters": {}
        }

    }
}


@router.get("/evaluations_payloads")
async def get_evaluations_payloads(

):
    return JSONResponse(status_code=200, content=jsonable_encoder(PAYLOAD, by_alias=False))


@router.get("/evaluations_project")
async def get_evaluation_project(
        is_active: bool = True,
        project: str = None,
        project_id: int = None,
        sort_by: str = None
):
    """
    Fetch evaluation project data. \n
    """
    # build filters
    filters = {
        "is_active": is_active,
        "project_name": project,
        "id": project_id,
        "status": sort_by,
    }

    evaluation_project_list = DatabaseHandler({'env': 'dev'}).get_evaluation_project(filters)

    return JSONResponse(status_code=200, content=jsonable_encoder(evaluation_project_list, by_alias=False))


@router.post("/evaluations_project")
async def save_project(
        project_data: UserEvaluationProject,
):
    """
    ProjectName, Model, System is required.

    Example: \n
    ```
    {
        "ProjectName": "scout_mixtral_agent",
        "Description": "scout mixtral agent ",
        "Status": "healthy",
        "Model": "mixtral_agent",
        "System": "scout",
        "Extra": "",
        "EmailSubscription": "",
        "NtAccount": "nvbot-evaluation"
    }
    ```
    """

    try:
        response = DatabaseHandler({'env': 'dev'}).add_evaluation_project(project_data)

    except Exception as ex:
        log.error(f"Error when save project, {ex}")
        return JSONResponse(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                            content=jsonable_encoder(f"Error: {ex}"))

    return JSONResponse(status_code=200, content=jsonable_encoder(response, by_alias=False))


@router.post("/dataset")
async def add_evaluation_dataset(
        column_map_str: str = '{"question":"Query", "reference":"Correct Answer", "additional_params": ["Correct Answer", "Required Citations", "Helpfulness Criteria"]}',
        dataset_file: UploadFile = File(...,
                                        description="Dataset file to be uploaded, accept excel format (xlsx), csv format (csv)"),
        judge_config: str = json.dumps(DEFAULT_JUDGE_PARAM),
        eval_type: Optional[str] = LLM_AS_A_JUDGE_EVALUATOR_TYPE,
        project_id: Optional[int] = None,
        dataset_name: Optional[str] = "",
        dataset_description: Optional[str] = None,
        storage_engine: Optional[str] = StorageType.DATASTORE.value,
        user_id: Optional[str] = NT_ACCOUNT_ID
) -> JSONResponse:
    """
    Column map is intended to take required column names for creating dataset, which includes \n
     - "question" - question initiated in conversation \n
     - "reference" - reference which will be used as ground truth when running llm-as-a-judge \n
     - [Optional] "additional_params" - which is a list, contains all column names which will be supplied in llm prompt as input variable \n
    expecting both question and reference column exist in Excel file uploaded.

    Example: \n
    ```
    {
        "Projectid": "2",
        "column_map": {
            "question": "Query",
            "reference": "Correct Answer",
            "additional_params": [
              "Required Citations",
              "Helpfulness Criteria",
              "Empathy Expected",
              "Short Answer Expected"
            ]
          },
    }
    ```

    judge_configs is used to generate lln-as-a-judge prompt with supplied additional variables, which will be embedded as curly brases wrapped camel_case terms,
    e,g: {require_citations} \n

    Example: \n
    ```
    {
            "prompt_module": "eval_prompt_library.metrics_eval_prompt.MetricsEvaluationPrompt",
            "output_format": "output_format",
            "template": "eval_template_v2",
            "scorers": [
              "Correctness Answer",
              "Helpfulness",
              "Empathy",
              "Conciseness"
            ]
    }
    ```
    """

    datastore_result = None
    file_name, file_extension = os.path.splitext(dataset_file.filename)
    file_name = convert_to_snake_case(file_name)
    file_fullname = f"{convert_to_snake_case(file_name)}{file_extension}"

    # Read the uploaded file
    contents = await dataset_file.read()
    db_records = []
    try:
        if contents:
            s3_folder_name = dataset_name
            if project_id and isinstance(project_id, int):
                evaluation_project = get_evaluation_project_by_id(project_id)
                if not dataset_name:
                    s3_folder_name = f"{evaluation_project.get('system')}_{evaluation_project.get('model')}"

            s3_response = S3DatasetHandler(dataset_name, {}). \
                upload_file_content(
                contents,
                f"{s3_folder_name}/dataset/{file_fullname}",
            )

            upload_status = s3_response.get("status")
            upload_filepath = s3_response.get("upload_filepath")
            print(f"Uploaded to s3: {upload_filepath}")
            assert upload_status.lower() == "success", f"failed to upload file to s3: {file_fullname}"
            datastore_result = {
                "Engine": StorageType.S3.value,
                "Name": s3_folder_name,
                "DatasetFolder": s3_folder_name,
                "DatasetPath": f"dataset/{file_fullname}"
            }
            db_records.append(datastore_result)

        if storage_engine.lower() == StorageType.DATASTORE.value:
            column_map = ColumnMapModel.parse_raw(column_map_str)

            # Initialize the DataFrame
            df = None

            # Check the file extension and read the file accordingly
            if file_extension == '.csv':
                df = pd.read_csv(pd.compat.StringIO(contents.decode('utf-8')), parse_dates=False)
            elif file_extension in ['.xls', '.xlsx']:
                df = pd.read_excel(pd.io.common.BytesIO(contents), parse_dates=False)

            evaluator_config = {
                "column_map": column_map.dict(),
                "model": {
                    "llm_name": REFERENCE_MODEL
                },
                "judge_config": json.loads(judge_config),
            }

            assert isinstance(evaluator_config['judge_config'], dict), "judge_config is not a dictionary"

            temp_folderpath = os.path.join(LOCAL_TMP_FOLDER, file_name)
            print(f"path: {temp_folderpath}")
            prepare_llm_as_a_judge_nemoeval_dataset(temp_folderpath, df, evaluator_config)

            # save_file_locally(file: UploadFile, destination: str):
            try:
                with open(os.path.join(temp_folderpath, DATASET_FILENAME), "wb") as buffer:
                    buffer.write(contents)
            except Exception as ex:
                logging.error(f"Failed to save to {temp_folderpath}: {ex}")

            # TODO: fetch project_name by id
            project_name = file_name

            # upload
            repo_name_unique = generate_unique_datastore_path(project_name, eval_type)
            print(f"nemo repo name: {repo_name_unique}")
            ds = NeMoDataStore(get_settings().NEMO_DS_URL, {
                "name": repo_name_unique,
                "upload_dir": temp_folderpath
            })
            ds.create_via_hfapi(
                path_in_repo=".")
            datastore_result = ds.datastore_content
            datastore_result["Engine"] = StorageType.DATASTORE.value
            db_records.append(datastore_result)

    except Exception as ex:
        log.error(f"Error when upload dataset, {ex}")
        return JSONResponse(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                            content=jsonable_encoder(f"Error: {ex}"))
    # remove from local
    try:
        if project_id is not None and datastore_result is not None:
            print(f"Added to dataset handler for dataset config: {datastore_result}")
            db_response = DatabaseHandler({'env': 'dev'}).add_evaluation_dataset(
                UserEvaluationDataset(
                    ProjectId=project_id,
                    Name=datastore_result.get("Name", ""),
                    Host=datastore_result.get("Engine", ""),
                    Description=dataset_description,
                    DatasetConfig=datastore_result,
                    NtAccount=user_id,
                )
            )
            print("Evaluation dataset id added: ", db_response)
            return JSONResponse(status_code=200, content=jsonable_encoder(
                {
                    "dataset_config": datastore_result,
                    "evaluation_dataset_id": db_response,
                }, by_alias=False))
        # shutil.rmtree(temp_folderpath, ignore_errors=True)
    except Exception as ex:
        log.error(f"Error when insert dataset data into database, {ex}")
        # return JSONResponse(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
        #                     content=jsonable_encoder(f"Error: {ex}"))

    return JSONResponse(status_code=200, content=jsonable_encoder(
        {
            "dataset_config": datastore_result,
        }, by_alias=False))
