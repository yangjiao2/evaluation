import http
import pprint

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

import base64
from typing import List, Optional, Dict, TypedDict
from pydantic import BaseModel, validator

from controllers.auth.auth_token_loader import AuthTokenLoader
from data_models.api.run_maker import RunMakerRequest, FlowConfigRequest, RunParamConfig
from nvbot_models.nvbot_platform_schemas.contracts.config_contracts.langchain_config.nv_langchain_config import \
    NVBotPlatformConfig
from nvbot_models.nvbot_platform_schemas.contracts.config_contracts.langchain_config.nvbot_langchain_config import \
    NVBotFlowConfig
from nvbot_models.request_models.bot_maker_request import BotMakerRequest, BotmakerRequestParameter

import json
import logging
import os
import uuid
import time
import requests

from configs.settings import get_settings, get_chatbot_url, get_graph_chatbot_url
from nvbot_models.request_models.fulfillment_request import FulfillmentRequest
from nvbot_utilities.utils import api_handler
from service_library.constants import NT_ACCOUNT_ID
from service_library.handler.config_loader import ConfigLoader
from service_library.handler.database_handler import DatabaseHandler
from service_library.run.parser_library.dict_parser import dict_parser
from service_library.run.parser_library.input_parser import prepare_chatbot_request, generate_fulfillment_request, \
    generate_orchestrator_fulfillment_request, format_chat_request_post_params
from service_library.run.parser_library.response_parser import fetch_post_chatbot_request_response

router = APIRouter(
    tags=["Adaptor"]
)

log = logging.getLogger('Completions Router')


class Message(BaseModel):
    role: str  # typically "user" or "system"
    content: str


class ExtraInfo(RunMakerRequest):
    project_id: Optional[int] = None
    env: Optional[str] = None
    system: Optional[str] = None
    model: Optional[str] = None
    project: Optional[str] = None
    userid: Optional[str] = None
    username: Optional[str] = NT_ACCOUNT_ID
    eval_id: Optional[str] = ""


class ChatRequest(BaseModel):
    model: str = ""
    messages: List[Message]
    max_tokens: Optional[int] = None
    temperature: Optional[int] = 0
    top_p: Optional[int] = 1
    stop: Optional[List[str]] = None
    # expect to take argument, such as project_id,
    user: Optional[str] = ""
    extra_body: Optional[ExtraInfo | dict]


class Choice(BaseModel):
    logprobs: Optional[dict]
    message: Message
    index: Optional[int]


class ChatResponse(BaseModel):
    id: str
    created: float
    # model: str
    choices: List[Choice]


class ChatRequestProcessing(BaseModel):
    messages: List[Message]
    extra_body: Optional[ExtraInfo | dict]
    output_schema: Optional[List[RunParamConfig]]


@router.post("/chat/completions_postprocessing")
async def completions_postprocessing(chat_request: ChatRequestProcessing):
    """
    Parsed output, this should be used for early testing on chatbot response. \n

    Orchestrator based chat completion example:
    1.
    ```
    {
  "model": "nvbot-adapter",
  "messages": [
    {
      "role": "system",
      "content": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
    },
    {
      "role": "user",
      "content": "How many devices with a role of 'pdu' at the site 'pdx01' do we have?"
    }
  ],
  "max_tokens": 25,
  "temperature": 0,
  "top_p": 0,
  "stop": null,
  "user": "",
  "extra_body": {
    "project_id": 13,
    "env": "sandbox",
    "system": "orchestrator_perceptor",
    "model": "mixtral_agent",
    "project": "perceptor_bot_qa",
    "userid": "nvbot-evaluation",
    "username": "",
    "eval_id": "eval-8Xs8abXj4sZ9G32n6Qpyxi"
    },
    "output_schema": [
        {
          "name": "Response",
          "type": "Attribute",
          "value": ""

        },
        {
          "name": "PostProcessResponse",
          "type": "Attribute",
          "value": "-1.post_process"

        },
        {
          "name": "PostProcessResponse",
          "type": "Function",
          "value": "service_library.run.parser_library.response_parser.process_orchestrator_node_data",
          "args": [
            "PostProcessResponse"
          ]

        },
        {
          "name": "Response",
          "type": "Function",
          "value": "service_library.run.parser_library.response_parser.get_orchestrator_response",
          "args": [
            "Response"
          ]
        }
    ]

}


    ```
    \n
    2.
    ```
    {
    "model": "nvbot-adapter",
    "messages": [
        {
            "role": "system",
            "content": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        },
        {
            "role": "user",
            "content": "What are NVIDIA core values?"
        }
    ],
    "max_tokens": 25,
    "temperature": 0,
    "top_p": 0,
    "stop": null,
    "user": "",
    "extra_body": {
        "project_id": 11,
        "env": "sandbox",
        "system": "nvinfo",
        "model": "mixtral_agent",
        "project": "eval_nvinfo_mixtral_agent_stg",
        "user_id": "nvbot-evaluation",
        "username": "",
        "eval_id": "eval-8Xs8abXj4sZ9G32n6Qpyxi"
    },
    "output_schema": [
       {
            "name": "Agent response",
            "type": "Attribute",
            "value": "0.agent"
        },
        {
            "name": "Call expert response",
            "type": "Attribute",
            "value": "-1.call_experts"
        },
        {
            "name": "Experts",
            "type": "Function",
            "value": "service_library.run.parser_library.response_parser.process_orchestrator_data",
            "args": [
                "Call expert response"
            ]
        },
        {
            "name": "Bot Answer",
            "type": "Function",
            "value": "service_library.run.parser_library.response_parser.get_orchestrator_response",
            "args": [
                "Agent response"
            ]
        }
    ]
    }

    ```
    """

    query_id = str(uuid.uuid4())
    print("Completions chat_request:", chat_request.dict())
    logging.info(f"Completions chat_request: {chat_request.dict()}")

    print(f"Completion extra_body param: {chat_request.extra_body}")

    questions = [msg.content for msg in chat_request.messages if msg.role.lower() == "user"]
    if len(questions) == 0:
        raise HTTPException(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                            detail="Invalid request: message does not have user content")

    chat_request_params = await format_chat_request_post_params(RunMakerRequest.model_validate(chat_request.extra_body
                                                                                          ), questions[0])

    system = chat_request_params.get("system")
    model = chat_request_params.get("model")
    env = chat_request_params.get("env")
    payload = chat_request_params.get("payload")
    chatbot_url = chat_request_params.get("chatbot_url")
    json_platform_config = chat_request_params.get("json_platform_config")
    chatbot_request = chat_request_params.get("chatbot_request")

    response, retry_count, duration_avg = await fetch_post_chatbot_request_response(chatbot_request, chatbot_url, env)

    nvbot_platform_config = NVBotPlatformConfig.model_validate(json_platform_config.model_dump())
    use_graph_chatbot_url = ('GraphConfig' in list(nvbot_platform_config.FlowConfig.dict().keys())) and (
            nvbot_platform_config.FlowConfig.GraphConfig is not None)
    if use_graph_chatbot_url:
        response = response.get("response", {})

    output_mapper = await dict_parser(parser_config=[schema.dict() for schema in chat_request.output_schema])

    if response:
        return output_mapper(response)

    return response


@router.post("/chat/completions")
async def completions(chat_request: ChatRequest):
    """
    Example:
    ```
    {
      "model": "string",
      "messages": [
        {
        "role": "user",
        "content": "hello"
        }
      ],
      "extra_body": {
        "env": "sandbox",
        "system": "nvinfo",
        "model": "mixtral_agent",
        "project": "eval_nvinfo_mixtral_agent_stg",
        "user_id": "nvbot-evaluation",
        "username": ""
      }
    }
    ```
    """
    # project = run_request.Project
    print("Completions chat_request:", chat_request.dict())
    logging.info(f"Completions chat_request: {chat_request.dict()}")

    print(f"Completion extra_body param: {chat_request.extra_body}")

    questions = [msg.content for msg in chat_request.messages if msg.role.lower() == "user"]
    if len(questions) == 0:
        raise HTTPException(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                            detail="Invalid request: message does not have user content")

    chat_request_params = await format_chat_request_post_params(RunMakerRequest.model_validate(chat_request.extra_body
                                                                                          ), questions[0])

    system = chat_request_params.get("system")
    model = chat_request_params.get("model")
    env = chat_request_params.get("env")
    payload = chat_request_params.get("payload")
    chatbot_url = chat_request_params.get("chatbot_url")
    json_platform_config = chat_request_params.get("json_platform_config")
    chatbot_request = chat_request_params.get("chatbot_request")

    try:
        # chatbot_url = get_settings().NVBOT_GRAPH_CHATBOT_URL
        logging.info(f"--> Question send to /chatbot: {chatbot_url}")
        print(f"--> Question send to /chatbot: {chatbot_url}")
        extra_body_dict = chat_request.extra_body
        chat_response, retry_count, duration_avg = await fetch_post_chatbot_request_response(chatbot_request, chatbot_url, env)

        response_text = chat_response.get("response_text")

        logging.info(f"Completion response_text: {response_text}")
        if response_text:
            return ChatResponse(
                id=str(chatbot_request.QueryId),
                created=time.time(),
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "logprobs": None,
                    "finish_reason": "stop"
                }]
            )

    except Exception as ex:
        log.error(f"Error when complete chat request, {ex}")
        return JSONResponse(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                            content=jsonable_encoder(f"Error when complete chat request: {ex}"))

    return JSONResponse(status_code=200, content=jsonable_encoder("No chat response received", by_alias=False))


@router.post("/chat/completions_preprocessing")
async def completions_preprocessing(chat_request: ChatRequest):
    """
    Example:
    ```
    {
      "model": "string",
      "messages": [
        {
        "role": "user",
        "content": "hello"
        }
      ],
      "extra_body": {
        "env": "sandbox",
        "system": "nvinfo",
        "model": "mixtral_agent",
        "project": "eval_nvinfo_mixtral_agent_stg",
        "user_id": "nvbot-evaluation",
        "username": ""
      }
    }
    ```
    """
    # project = run_request.Project
    query_id = str(uuid.uuid4())
    print("Completions chat_request:", chat_request.dict())
    logging.info(f"Completions chat_request: {chat_request.dict()}")

    print(f"Completion extra_body param: {chat_request.extra_body}")

    questions = [msg.content for msg in chat_request.messages if msg.role.lower() == "user"]
    if len(questions) == 0:
        raise HTTPException(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                            detail="Invalid request: message does not have user content")

    chat_request_params = await format_chat_request_post_params(RunMakerRequest.model_validate(chat_request.extra_body
                                                                                          ), questions[0])

    system = chat_request_params.get("system")
    model = chat_request_params.get("model")
    env = chat_request_params.get("env")
    payload = chat_request_params.get("payload")
    chatbot_url = chat_request_params.get("chatbot_url")
    json_platform_config = chat_request_params.get("json_platform_config")
    chatbot_request = chat_request_params.get("chatbot_request")

    try:
        # query_id = str(uuid.uuid4())

        # chatbot_url = get_settings().NVBOT_GRAPH_CHATBOT_URL
        logging.info(f"--> Question send to /chatbot: {chatbot_url}")
        extra_body_dict = chat_request.extra_body
        chat_response, retry_count, duration_avg = await fetch_post_chatbot_request_response(chatbot_request, chatbot_url, env)
        pprint.pprint(chat_response)
        return JSONResponse(status_code=200, content=jsonable_encoder(chat_response))

    except Exception as ex:
        log.error(f"Error when complete chat request, {ex}")
        return JSONResponse(status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                            content=jsonable_encoder(f"Error when complete chat request: {ex}"))

    return JSONResponse(status_code=200, content=jsonable_encoder("No chat response received", by_alias=False))
