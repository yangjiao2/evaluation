import logging
import os

from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Optional

# from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings

from nvbot_sdk.utils.cache.redis_secrets import RedisSecrets
from nvbot_utilities.utils.caching import Cache
from nvbot_utilities.utils.utilities import read_yaml, get_allowed_dls
from nvbot_utilities.utils.vault.nv_vault import NVVault
from nvbot_utilities.utils.starfleet.starfleet_models import StarfleetConfig
from nvbot_utilities.utils.starfleet.constants import LANGCHAIN_API_SCOPE

from openai import OpenAI

env = os.getenv("ENV")

logger = logging.getLogger(__name__)

def _read_config():
    """
    Read the configuration file
    @return: config
    """
    # Read the config file
    base_path = Path(__file__).parent
    config_path = Path.joinpath(base_path, "config.yaml")
    cfg = read_yaml(config_path)
    return cfg


def build_starfleet_secrets(data: dict) -> StarfleetConfig:
    # secrets: StarfleetConfig = StarfleetConfig.model_validate(data)
    secrets: StarfleetConfig = StarfleetConfig.model_validate(data)
    scope_info: str = LANGCHAIN_API_SCOPE
    secrets.scopes = scope_info.split(" ")

    return secrets


def _get_secrets(env: str = os.getenv("ENV")):
    cfg = _read_config()
    secrets = NVVault()
    # print(f"reading {env} env variables")

    # AWS S3 Secrets
    aws = secrets.read_secret(cfg["aws"]["path"])
    cfg["aws"].update(aws)

    # SNOW Service Account
    snow_secrets = secrets.read_secret(cfg["svc_jarvis"]["path"])
    cfg["svc_jarvis"].update(snow_secrets)

    # Redis secrets
    redis_secrets = secrets.read_secret(cfg["redis"]["path"])
    cfg["redis"].update(redis_secrets)

    # Helios secrets
    helios_secrets = secrets.read_secret(cfg["helios"]["path"])
    cfg["helios"].update(helios_secrets)

    # Mulesoft secrets
    mulesoft_secrets = secrets.read_secret(cfg["mulesoft"]["path"].format(env))
    cfg["mulesoft"].update(
        {
            "client_id": mulesoft_secrets.get("client_id"),
            "client_secret": mulesoft_secrets.get("client_secret"),
            "mulesoft_url": mulesoft_secrets.get("mulesoft_url"),
            "pre_prod_base_url": mulesoft_secrets.get("pre_prod_base_url"),
            "snow_live_agent_count_url": mulesoft_secrets.get("snow_live_agent_count_url"),
            "singleuser_url": mulesoft_secrets.get("singleuser_url"),
        }
    )

    # NVBot related secrets
    nvboturls_secrets = secrets.read_secret(cfg.get("nvboturls").get("path").format(env))
    cfg["nvboturls"].update(nvboturls_secrets)

    # for chat completion request
    cfg["nvbotchatboturls"] = {
        "sandbox": secrets.read_secret(cfg.get("nvboturls").get("path").format("sandbox"))['nvbotchatbot'],
        "dev": secrets.read_secret(cfg.get("nvboturls").get("path").format("dev"))['nvbotchatbot'],
        "stg": secrets.read_secret(cfg.get("nvboturls").get("path").format("stg"))['nvbotchatbot'],
        # hard code since prod url is not stored in secret
        "prd": secrets.read_secret(cfg.get("nvboturls").get("path").format("stg"))['nvbotchatbot'].replace('//stgbot',
                                                                                                           '//nvbot',
                                                                                                           1),
    } if env.lower() != "prd" else {
        "prd": secrets.read_secret(cfg.get("nvboturls").get("path").format("prd"))['nvbotchatbot']
    }

    cfg["nvbotgraphchatbotrurls"] = {
        "sandbox": secrets.read_secret(cfg.get("nvboturls").get("path").format("sandbox"))['graphorchestrator'],
        "dev": secrets.read_secret(cfg.get("nvboturls").get("path").format("dev"))['graphorchestrator'],
        "stg": secrets.read_secret(cfg.get("nvboturls").get("path").format("stg"))['graphorchestrator'],
        # hard code since prod url is not stored in secret
        "prd": secrets.read_secret(cfg.get("nvboturls").get("path").format("stg"))['graphorchestrator'].replace(
            '//stgbot',
            '//nvbot', 1),
    } if env.lower() != "prd" else {
        "prd": secrets.read_secret(cfg.get("nvboturls").get("path").format("prd"))['graphorchestrator']
    }

    cfg["nvbotconfigurls"] = {
        "sandbox": secrets.read_secret(cfg.get("nvboturls").get("path").format("sandbox"))['nvbotconfig'],
        "dev": secrets.read_secret(cfg.get("nvboturls").get("path").format("dev"))['nvbotconfig'],
        "stg": secrets.read_secret(cfg.get("nvboturls").get("path").format("stg"))['nvbotconfig'],
        # hard code since prod url is not stored in secret
        "prd": secrets.read_secret(cfg.get("nvboturls").get("path").format("stg"))['nvbotconfig'].replace('//stgbot',
                                                                                                          '//nvbot', 1),
    } if env.lower() != "prd" else {
        "prd": secrets.read_secret(cfg.get("nvboturls").get("path").format("prd"))['nvbotconfig']
    }

    # open ai api secrets
    open_ai_secrets = secrets.read_secret(cfg["openai"]["path"])
    cfg["openai"].update(open_ai_secrets)

    # nvbot allowed dls
    allowed_dls = get_allowed_dls(env)
    cfg["allowed_dls"] = allowed_dls

    # # sales collection list
    # collection_list = cfg["sales_collection_list"][env]
    # cfg["sales_collection_list"] = collection_list

    # get starfleet client ids
    client_ids = secrets.read_secret(cfg["starfleet"]["clients"].format(env))
    cfg["starfleet_clients"] = client_ids

    # starfleet secrets
    starfleet_urls = secrets.read_secret(cfg["starfleet"]["urls"].format(env))
    cfg["starfleet"].update(starfleet_urls)

    # Sendgrid token
    sendgrid_secrets = secrets.read_secret(cfg['email']['path'])
    cfg['email'].update(sendgrid_secrets)

    langchain_client_id = client_ids.get("langchain")
    langchain_secrets = secrets.read_secret(cfg["starfleet"]["ssa_path"].format(langchain_client_id))
    cfg["langchain"] = langchain_secrets | starfleet_urls

    sensitvity_client_id = client_ids.get("sensitivity-classifier")
    sensitivity_secrets = secrets.read_secret(cfg["starfleet"]["ssa_path"].format(sensitvity_client_id))
    cfg["sensitivity"] = sensitivity_secrets | starfleet_urls

    # if env not in ["dev", "sandbox"]:
    #     llm_gateway_secrets = secrets.read_secret(cfg["llm_gateway_prd"]["path"])
    #     cfg["llm_gateway"].update({"secrets": LLMSecrets.model_validate(llm_gateway_secrets)})
    #
    #     llm_list = secrets.read_secret(cfg["llm_list_prd"]["path"])
    #     cfg["llm_list"].update({"models": ModelList.model_validate(llm_list)})
    # else:
    #     llm_gateway_secrets = secrets.read_secret(cfg["llm_gateway"]["path"])
    #     cfg["llm_gateway"].update({"secrets": LLMSecrets.model_validate(llm_gateway_secrets)})
    #
    #     llm_list = secrets.read_secret(cfg["llm_list"]["path"])
    #     cfg["llm_list"].update({"models": ModelList.model_validate(llm_list)})
    if env not in ["dev", "sandbox"]:
        llm_gateway_secrets = secrets.read_secret(cfg["llm_gateway_prd"]["path"])
        cfg["llm_gateway"].update({"secrets": LLMSecrets.model_validate(llm_gateway_secrets)})

        llm_list = secrets.read_secret(cfg["llm_list_prd"]["path"])
        cfg["llm_list"].update({"models": ModelList.model_validate(llm_list)})
    else:
        llm_gateway_secrets = secrets.read_secret(cfg["llm_gateway"]["path"])
        cfg["llm_gateway"].update({"secrets": LLMSecrets.model_validate(llm_gateway_secrets)})

        llm_list = secrets.read_secret(cfg["llm_list"]["path"])
        cfg["llm_list"].update({"models": ModelList.model_validate(llm_list)})

    # if True:  # env not in ["dev", "sandbox"]:
    #     cfg["nemo_url"] = {
    #         "datastore": "https://datastore.stg.llm.ngc.nvidia.com",
    #         "evaluator": "https://evaluation.stg.llm.ngc.nvidia.com"
    #     }
    # else:
    #     cfg["nemo_url"] = {
    #         "datastore": "https://datastore.dev.llm.ngc.nvidia.com",
    #         "evaluator": "https://evaluation.dev.llm.ngc.nvidia.com"
    #     }
    nvcf_secrets = secrets.read_secret(cfg["nvcf_keys"]["path"])
    cfg["nvcf_keys"].update(nvcf_secrets)
    if env in ['stg', 'prd']:
        cfg["nvcf_keys"]["nvcf_key"] = nvcf_secrets.get("private")
    else:
        cfg["nvcf_keys"]["nvcf_key"] = nvcf_secrets.get("public")

    nvbugs_secrets = secrets.read_secret(cfg["nvbugs"]["path"])
    cfg["nvcf_keys"]["nvdev_key"] = nvbugs_secrets.get("NVDEV_API_KEY")

    datadog_secrets = secrets.read_secret(cfg["datadog"]["path"])
    cfg["datadog"].update(datadog_secrets)

    if env in ['prd']:
        cfg["auth"].update({
            "nonprd": {},
            "prd": secrets.read_secret(f'{cfg["auth"]["path"]}/prd'),
        })
    else:
        cfg["auth"].update({
            "nonprd": secrets.read_secret(f'{cfg["auth"]["path"]}/nonprd') if env.lower() not in ['prd'] else {},
            "prd": secrets.read_secret(f'{cfg["auth"]["path"]}/prd'),
        })

    logger.info(f"Variable configuration loaded successfully for {env} env")
    print(f"Variable configuration loaded successfully for {env} env.")


    return cfg


def _build_redis_secret(redis_config: dict):
    redis_secrets = {
        "redis_url": redis_config["server"],
        "password": redis_config["password"],
        "username": redis_config["username"],
    }
    logger.info(f"Redis loaded successfully: {redis_config['server']}")
    return redis_secrets


class LLMSecrets(BaseModel):
    chat_completions_endpoint: str
    client_id: str
    client_secret: str
    completions_endpoint: str
    grant_type: str
    scope: str
    token_endpoint: str
    nemo_token: str
    nemo_tokenizer_endpoint: str


class ModelInfo(BaseModel):
    model: str
    endpoint: str = None
    model_config = ConfigDict(protected_namespaces=())


class ModelSecrets(BaseModel):
    model_name: str
    model_details: ModelInfo
    # The following to remove UserWarning:
    # Field "model_name" in ModelSecrets has conflict with protected namespace "model_".
    model_config = ConfigDict(protected_namespaces=())


class ModelList(BaseModel):
    models: List[ModelSecrets]
    model_config = ConfigDict(protected_namespaces=())


class Settings(BaseSettings):
    """
    Read and update config required for the app
    """

    config: dict = _get_secrets()

    ## NVBOT
    NVBOT_FULFILLMENT_URL: str = config["nvboturls"]["nvbotfulfillments"]
    # NVBOT_FULFILLMENT_URL: str = "http://localhost:5000/fulfillment/"
    NVBOT_SERVICES_URL: str = config["nvboturls"]["nvbotservicesurl"]
    # NVBOT_SERVICES_URL: str = "http://localhost:4000/services"
    NVBOT_CHATBOT_URL: str = config['nvboturls']['nvbotchatbot']
    # NVBOT_CHATBOT_URL: str = "http://localhost:8000/langchain/chatbot"
    NVBOT_GRAPH_CHATBOT_URL: str = config['nvboturls']['graphorchestrator']
    # NVBOT_GRAPH_CHATBOT_URL: str = "http://localhost:12000/graph/orchestrate"

    NVBOT_CONFIG_SELECTOR_URL: str = config['nvboturls']['nvbotconfig']
    # NVBOT_CONFIG_SELECTOR_URL: str = "http://127.0.0.1:7003/configs/"
    NVBOT_EVALUATION_ADAPTOR_URL: str = "https://devbot-api.nvidia.com/evaluation/chat/completions"
    NVBOT_EVALUATION_URL: str = "https://devbot-api.nvidia.com/evaluation"
    NVBOT_EVALUATION_UI_URL: str = "https://nvbot-evaluation-dev.nvidia.com"

    NVBOT_CHATBOT_URLS: dict = config['nvbotchatboturls']
    NVBOT_GRAPH_CHATBOT_URLS: dict = config['nvbotgraphchatbotrurls']
    NVBOT_CONFIG_SELECTOR_URLS: dict = config['nvbotconfigurls']

    HELIOS_API_KEY: str = config["helios"]["token"]
    REDIS_SECRETS: dict = _build_redis_secret(config["redis"])
    REDIS_CACHE_SECRET: RedisSecrets = RedisSecrets.model_validate(config['redis']) # RedisSecrets.model_validate(config['redis'])

    MULESOFT_URL: str = config["mulesoft"]["mulesoft_url"]
    MULESOFT_CLIENT_ID: str = config["mulesoft"]["client_id"]
    MULESOFT_CLIENT_SECRET: str = config["mulesoft"]["client_secret"]

    ## LLM
    OPENAI_API_KEY: str = config["openai"]["api_key"]
    AGENT_MODEL: List[str] = config["agent"]["model"]
    AGENT_TYPE: str = config["agent"]["type"]
    OPENDOMAIN_MODEL: List[str] = config["opendomain"]["model"]
    LLM_GATEWAY_SECRETS: LLMSecrets = config["llm_gateway"]["secrets"]
    LLM_LIST: ModelList = config["llm_list"]["models"]

    ## SYSTEM
    CACHE_ANSWER_TIME: int = config["cache_answer_time"]
    SINGLE_USER_URL: str = config["mulesoft"]["singleuser_url"]
    ALLOWED_DL_LIST: Dict[str, list[str]] = config["allowed_dls"]

    ## STARFLEET
    STARFLEET_SECRETS: StarfleetConfig = build_starfleet_secrets(config["langchain"])
    SENSITIVITY_DETECTION: bool = True
    SENSITIVITY_DEBUG_FLAG: bool = False  # added for ITNVBOT-2447
    SENSITIVITY_API_URL: str = "https://llm-classifier-stg.nvidia.com/v1/dlp-decision"
    SENSITIVITY_API_HOST: str = "10.10.0.1"
    SENSITIVITY_SECRETS: StarfleetConfig = build_starfleet_secrets(config["sensitivity"])
    LANGCHAIN_DEBUG_MODE: bool = False
    HIDE_STATUS_DETAIL: dict = config["hidedetailstatus"]

    ## AUTH
    NVSF_CLIENT_IDS: dict = {
        "nonprd": config["auth"]["nonprd"].get("client_id"),
        "prd": config["auth"]["prd"].get("client_id")
    }
    NVSF_SUBS: dict = {
        "nonprd": config["auth"]["nonprd"].get("sub"),
        "prd": config["auth"]["prd"].get("sub")
    }

    # TODO: add to NVVault
    NVSF_TOKEN_URLS: dict = {
        "nonprd": "https://stg.login.nvidia.com/token",
        "prd": "https://login.nvidia.com/token"
    }
    NVSF_AUTHORIZE_URLS: dict = {
        "nonprd": "https://stg.login.nvidia.com/device/authorize",
        "prd": "https://login.nvidia.com/device/authorize"
    }
    NVSF_CLIENT_TOKEN_URLS: dict = {
        "nonprd": "https://stg.login.nvidia.com/client_token",
        "prd": "https://login.nvidia.com/client_token",
    }

    SENDGRID_TOKEN: str = config['email']['token']

    # NeMo Service
    # NEMO_EVAL_URL: str = config['nemo_url']['evaluator']
    # NEMO_DS_URL: str = config['nemo_url']['datastore']

    # self-host in cluster
    NEMO_EVAL_URL: str = "http://evaluation-ms-prod-nemo-evaluator.nemo-evaluation.svc.cluster.local:7331"
    NEMO_EVAL_URL: str = "http://devbot-nemo-evaluator-api.nvidia.com"  # "http://localhost:7331"

    NEMO_DS_URL: str = "http://nemo-datastore.nemo-evaluation.svc.cluster.local:8000"
    NEMO_DS_URL: str = "http://devbot-nemo-datastore-api.nvidia.com"  # "http://localhost:8000"

    ## Datadog dev
    DATADOG_API_KEY: str = config["datadog"]["api_key"]
    DATADOG_APP_KEY: str = config["datadog"]["app_key"]
    DATADOG_HOST: str = config["datadog"]["host"]

    ## NVCF key
    NVCF_API_KEY: str = config["nvcf_keys"]["nvcf_key"]
    PUBLIC_NVCF_API_KEY: str = config["nvcf_keys"]["public"]
    PRIVATE_NVCF_API_KEY: str = config["nvcf_keys"]["private"]
    NVDEV_API_KEY: str = config["nvcf_keys"]["nvdev_key"]



@lru_cache()
def get_settings():
    return Settings()


@lru_cache
def get_cache_session(key_hash: Optional[str] = None):
    settings = get_settings()
    if key_hash:
        return Cache(redis_secrets=settings.REDIS_SECRETS, cache_hash=key_hash)
    return Cache(redis_secrets=settings.REDIS_SECRETS)


def get_chatbot_url(env: str = os.getenv("ENV")):
    chatbot_url = get_settings().NVBOT_CHATBOT_URLS.get(env, "CHATBOT_URL")
    # return get_settings().NVBOT_CHATBOT_URL
    print(f"/chatbot: {chatbot_url}")
    return chatbot_url


def get_graph_chatbot_url(env: str = os.getenv("ENV")):
    chatbot_url = get_settings().NVBOT_GRAPH_CHATBOT_URLS.get(env, "GRAPH_CHATBOT_URLS")
    # return get_settings().NVBOT_GRAPH_CHATBOT_URL
    print(f"/orchestrate: {chatbot_url}")
    return chatbot_url


def get_config_url(env: str = os.getenv("ENV")):
    config_url = get_settings().NVBOT_CONFIG_SELECTOR_URLS.get(env, "CONFIG_SELECTOR_URL")
    # return get_settings().NVBOT_CONFIG_SELECTOR_URL
    print(f"/config: {config_url}")
    return config_url
