import http
import logging
import uuid

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from controllers.auth.auth_token_loader import AuthTokenLoader

from configs.settings import get_graph_chatbot_url, get_chatbot_url
from data_models.api.run_maker import RunMakerRequest
from nvbot_models.nvbot_platform_schemas.contracts.config_contracts.langchain_config.nv_langchain_config import \
    NVBotPlatformConfig
from nvbot_models.request_models.fulfillment_request import FulfillmentRequest
from nvbot_utilities.utils.utilities import get_class_module, create_func_instance
from service_library.constants import NT_ACCOUNT_ID
from service_library.handler.database_handler import DatabaseHandler
from service_library.nemo_ms.nemo_service_helper import get_extract_run_maker_request
from service_library.utils.configuration_helper import get_platform_config, get_evaluation_config, \
    get_evaluation_project_by_id
from service_library.utils.logging import log_errors
from service_library.utils.request_helpers import create_header
from service_library.utils.run_helpers import generate_short_uuid


@log_errors("Format post params")
async def format_chat_request_post_params(run_request: RunMakerRequest, question: str, enable_logging: bool = True):
    extra_body_dict = get_extract_run_maker_request(run_request)

    evaluation_history_id = extra_body_dict.get("evaluation_history_id") if extra_body_dict else None
    project_id = extra_body_dict.get("project_id") if extra_body_dict else None
    env = extra_body_dict.get("env") if extra_body_dict else "dev"
    system = extra_body_dict.get("system") if extra_body_dict else None
    model = extra_body_dict.get("model") if extra_body_dict else None
    userid = NT_ACCOUNT_ID # extra_body_dict.get("user_id") or
    configid = extra_body_dict.get("config_id") if extra_body_dict else None
    project = extra_body_dict.get("project") if extra_body_dict else None

    json_platform_config = run_request.PlatformConfig
    json_eval_config = run_request.EvaluationConfig

    if evaluation_history_id:
        # fetch db
        evaluation_history = DatabaseHandler({'env': 'dev'}).get_evaluation_history(int(evaluation_history_id))
        if not run_request.PlatformConfig:
            json_platform_config = evaluation_history.get('flowconfig_metadatajson', None)
        if not run_request.EvaluationConfig:
            json_eval_config = evaluation_history.get('evaluation_metadatajson', None)
    try:
        if json_platform_config is None or json_eval_config is None:
            # fetch db
            run_request = RunMakerRequest(
                UserId=userid,
                Env=env,
                System=system,
                Model=model,
                ProjectId=project_id,
                Project=project,
                ConfigId=configid,
            )
            evaluation_project = get_evaluation_project_by_id(run_request.ProjectId)
            if evaluation_project is not None and system is None:
                system = evaluation_project.get("system") if system is None else evaluation_project.get("system")
            if evaluation_project is not None and model is None:
                model = evaluation_project.get("model") if model is None else evaluation_project.get("model")

        if json_platform_config is None:
            json_platform_config = await get_platform_config(run_request, evaluation_project)
        if json_eval_config is None and evaluation_project is not None:
            json_eval_config = await get_evaluation_config(run_request, evaluation_project)

        assert json_platform_config is not None, f"Failed to get nvbot platform config by {extra_body_dict}"
        nvbot_platform_config = NVBotPlatformConfig.model_validate(json_platform_config.model_dump())


    except Exception as ex:
        err_message = f"Error when fetch bot config, {ex}"
        logging.error(err_message)
        return HTTPException(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                             detail=jsonable_encoder(err_message))

    use_graph_chatbot_url = ('GraphConfig' in list(nvbot_platform_config.FlowConfig.dict().keys())) and (
            nvbot_platform_config.FlowConfig.GraphConfig is not None)
    chatbot_url = get_graph_chatbot_url(env) if use_graph_chatbot_url else get_chatbot_url(env)

    query_id = str(uuid.uuid4())
    if use_graph_chatbot_url:
        request = generate_orchestrator_fulfillment_request(question, query_id, system=system, model=model,
                                                            user_id=userid)
    else:
        request = generate_fulfillment_request(question, query_id, system=system, model=model,
                                               user_id=userid)

    logging.info(f"{system}, {model} Completion question: {question}")
    print(f"{system}, {model} with question: `{question}`")
    chatbot_request = prepare_chatbot_request(request,
                                              system,
                                              model,
                                              None,
                                              json_platform_config,
                                              enable_logging)

    return {
        "chatbot_url": chatbot_url,
        "env": env,
        "system": system,
        "model": model,
        "configid": configid,
        "payload": request,
        "json_platform_config": json_platform_config,
        "json_eval_config": json_eval_config,
        "chatbot_request": chatbot_request
    }


@log_errors('Prepare chatbot request')
def prepare_chatbot_request(
        request_data: dict,
        system,
        model,
        attachment=None,
        platform_config=dict,
        enable_logging=True) -> FulfillmentRequest:
    # request_data = generate_fulfillment_request(**request_data)
    chatbot_request = FulfillmentRequest.model_validate(request_data)
    chatbot_request.System = system
    chatbot_request.Model = model
    chatbot_request.Attachments = attachment
    chatbot_request.PlatformConfig = NVBotPlatformConfig.model_validate(platform_config.model_dump())
    # enforce logging to get output
    # chatbot_request.PlatformConfig.FlowConfig.BotConfig.EnableAnswerLogging = enable_logging
    return chatbot_request


def generate_orchestrator_fulfillment_request(search_query="", session_id = "", query_id="123456789", system="", model="", attachment=[],
                                              user_id=NT_ACCOUNT_ID):

    if not session_id:
        session_id = generate_short_uuid()
    print (f"QueryId: {query_id}, SessionId: {session_id}, Query: {search_query}")
    return {
        "Query": search_query,
        "Intent": "irqa",
        "Domain": "evaluator",
        "UserId": user_id,
        "SessionId": session_id,
        "QueryId": query_id,
        "Parameters": {
            "NvidiaID": user_id,
            "Source": system,
            "SourceType": "private",
            "Tz": "America/Los_Angeles",
            "TzLabel": "Pacific Daylight Time",
            "TzOffset": -25200,
            "IsTest": "false",
            "IsStream": False,
        },
        "System": system,
        "Model": model,
        "Attachments": attachment,
    }


def generate_fulfillment_request(search_query="", session_id = "", query_id="123456789", system="", model="", attachment=[],
                                 user_id=NT_ACCOUNT_ID):
    if not session_id:
        session_id = generate_short_uuid()
    print(f"QueryId: {query_id}, SessionId: {session_id}, Query: {search_query}")
    return {
        "Query": search_query,
        "Intent": "irqa",
        "Domain": "evaluator",
        "UserId": user_id,
        "SessionId": session_id,
        "QueryId": query_id,
        "Parameters": {
            "NvidiaID": user_id,
            "Source": system,
            "SourceType": "private",
            "Tz": "America/Los_Angeles",
            "TzLabel": "Pacific Daylight Time",
            "TzOffset": -25200,
            "IsTest": "false",
        },
        "System": system,
        "Model": model,
        "Attachments": attachment,
    }


@log_errors('Request parser')
def request_parser(parser_config: dict):
    parser_config = parser_config

    def func(row: dict):
        result = {}
        for config in parser_config:
            name = config["name"]
            type = config["type"]
            value = config["value"]

            if type.lower() == "text":
                # result = row
                result[value] = row[name]
            elif type.lower() == 'attribute':
                obj = row
                attributes = value.split(".")
                for attr in attributes:
                    if obj.get(attr, None):
                        obj = obj[attr]
                    else:
                        continue
                if name:
                    result[name] = obj
                else:
                    result = obj

            elif type.lower() == "function":
                parameters = config["args"]
                kwargs = [row[key] for key in parameters]

                module_name, function_name = get_class_module(value)
                func = create_func_instance(module_name=module_name, class_name=function_name)

                if name:
                    result[name] = func(*kwargs)
                else:
                    result = func(*kwargs)

        return result

    return func
