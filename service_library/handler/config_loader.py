import os
import boto3
from datetime import datetime
from typing import Optional
import json
import logging
import uuid
import aiohttp
from nvbot_utilities.utils import api_handler

from configs.settings import get_settings, get_cache_session, get_config_url
from data_models.api.run_maker import RunMakerRequest, NVBotEvaluationConfig, EvaluationSchema, FlowConfigRequest
from service_library.constants import NT_ACCOUNT_ID
from service_library.utils.logging import log_errors

from nvbot_models.nvbot_platform_schemas.contracts.config_contracts.langchain_config.nv_langchain_config import (
    NVBotPlatformConfig)
from configs.settings import get_settings, get_cache_session
from controllers.auth.auth_token_loader import AuthTokenLoader

from service_library.utils.request_helpers import create_header

logger = logging.getLogger(__name__)


class ConfigLoader:
    """A loader to help retrieve config flow."""

    def __init__(self, env: str = "dev"):
        _settings = get_settings()
        self.datasets = {}
        self.env = env
        self.cache_session = get_cache_session()
        self.auth_token = AuthTokenLoader(env).token

    @log_errors('fetch flow model config')
    async def get_flow_model_config(self, request: FlowConfigRequest) -> Optional[dict]:
        settings = get_settings()
        if request.ConfigId:
            logger.info(f'Fetched flow config for BotId {request.ConfigId}')
            try:
                config_url = f"{get_config_url(self.env)}bot/{request.ConfigId}"
                logging.info(f"--> Load flow config /configs/bot/ at: {config_url}")
                print('Loading config at --> ', config_url)
                data = {
                    'user_name': request.UserId,
                    'bot_id': request.ConfigId
                }

                headers = create_header(self.auth_token)
                # response = api_handler.get_data(url=configurl, headers=headers)

                async with aiohttp.ClientSession() as request_session:
                    async with await request_session.get(config_url, headers=headers, params=data) as resp:
                        assert resp.status == 200, f"Fail to fetch flow config: {json.dumps(data)}"
                        # if resp.status == 200:
                        bot_config = await resp.json()
                        config = bot_config.get("FlowConfig")
                        if bot_config and config:
                            return bot_config

            except Exception as ex:
                logger.error(f"Exception in load flow model config from config manager via botId: {ex}")
                return None
        elif request.System and request.Model:
            logger.info(f'Fetched flow config for {request.System} and model {request.Model}')
        try:
            config_url = f"{get_config_url(self.env)}langchainconfig"
            logging.info(f"--> Load flow config /langchainconfig at: {config_url}")
            print ('Loading config at --> ', config_url)
            data = {
                'user_name': request.UserId,
                'system': request.System,
                'model': request.Model if isinstance(request.Model, str) else request.Model.value,
            }

            headers = create_header(self.auth_token)
            # response = api_handler.get_data(url=configurl, headers=headers)
            async with aiohttp.ClientSession() as request_session:
                async with await request_session.get(config_url, headers=headers, params=data) as resp:
                    assert resp.status == 200, f"Fail to fetch flow config: {json.dumps(data)}"
                    # if resp.status == 200:
                    return await resp.json()
        except Exception as ex:
            logger.error(f"Exception in load flow model config from config manager via system: {ex}")
            return None

    @log_errors('fetch bot config')
    async def get_bot_config(self, request: FlowConfigRequest) -> Optional[dict]:
        settings = get_settings()
        logger.info(f'Fetched flow config for {request.System} and model {request.Model}')
        try:
            url = f"{get_config_url(self.env)}botconfig"
            print ('Loading bot config at --> ', url)
            data = {
                'username': request.UserId or NT_ACCOUNT_ID,
                'system': request.System,
                # 'model': request.Model.value,
                # 'attachment_type': request.Attachments[0].Type if request.Attachments and len(request.Attachments) > 0 else "",
            }

            headers = create_header(self.auth_token)

            async with aiohttp.ClientSession() as request_session:
                async with await request_session.get(url, headers=headers, params=data) as resp:
                    assert resp.status == 200, f"Fail to fetch bot config: {json.dumps(data)}"
                    # if resp.status == 200:
                    return await resp.json()
        except Exception as ex:
            logger.error(f"Exception in load bot config from config manager:\n{ex}")
            return None


    @log_errors('fetch evaluation config')
    async def get_evaluation_schema(self, config_name: str) -> Optional[dict]:
        try:
            if config_name in ["graph_evaluation_schema", "evaluation_schema", "custom_evaluation", "llm_as_a_judge"]:
                path = os.path.join("asset", "template", f"{config_name}.json")
                with open(path, 'r') as json_file:
                    json_eval_config = json.load(json_file)

                return NVBotEvaluationConfig.model_validate(json_eval_config)

            else:
                path = os.path.join("asset", f"{config_name}.json")

                assert os.path.isfile(path), f"File does not exist: {path}"
                with open(path, 'r') as json_file:
                    json_eval_config = json.load(json_file)

                return json_eval_config

        except Exception as ex:
            logger.error(f"Exception in load eval config {config_name} from config manager:\n{ex}")
            return None



