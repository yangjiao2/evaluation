import json
import os

import requests
from configs.settings import get_settings


def get_oauth_client_token():
    client_id = get_settings().LLM_GATEWAY_SECRETS.client_id
    client_secret = get_settings().LLM_GATEWAY_SECRETS.client_secret
    token_url = get_settings().LLM_GATEWAY_SECRETS.token_endpoint
    scope = get_settings().LLM_GATEWAY_SECRETS.scope

    assert client_id, f"Azure stg gateway client_id:{client_id} is None"
    assert client_secret, f"Azure stg gateway client_secret:{client_id} is None"

    headers = {"Content-Type": "application/json"}
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": scope,
        "grant_type": "client_credentials"
    }

    response = requests.post(token_url, headers=headers, json=data)
    api_key = json.loads(response.text)["access_token"]
    return api_key


def get_prd_oauth_client_token():
    client_id = os.getenv('LLM_GATEWAY_CLIENT_ID')
    client_secret = os.getenv('LLM_GATEWAY_CLIENT_SECRET')
    print("client_id:", client_id)
    print("client_secret:", client_secret)

    assert client_id, f"Azure prod gateway client_id:{client_id} is None"
    assert client_secret, f"Azure prod gateway client_secret:{client_id} is None"

    token_url = "https://prod.api.nvidia.com/oauth/api/v1/ssa/default/token"
    scope = get_settings().LLM_GATEWAY_SECRETS.scope

    headers = {"Content-Type": "application/json"}
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": scope,
        "grant_type": "client_credentials"
    }

    response = requests.post(token_url, headers=headers, json=data)
    api_key = json.loads(response.text)["access_token"]
    return api_key
