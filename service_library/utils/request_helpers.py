import json
import logging
import os
from typing import Any, Dict, Optional
import requests
from enum import Enum

from configs import settings
from configs.settings import get_cache_session, get_settings
from controllers.auth.auth_token_loader import AuthTokenLoader

from service_library.constants import NT_ACCOUNT_ID
from service_library.handler.utils import get_person_data
from service_library.utils.logging import log_errors
from datetime import datetime

class RequestType(Enum):
    GET = 'GET'
    POST = 'POST'
    DELETE = 'DELETE'


@log_errors("Evict JWKs key")
def evict_jwk_token(env: str):

    cache_key = 'jwks_keys'
    key_hash = f'jwks_keys:{env}:sso'
    cache_session = settings.get_cache_session(key_hash)
    # self.cache.hget(f'jwks_keys:{env}:sso', 'jwks_keys')
    # cache.update_expire_cache(cache_key, value=json.dumps(jwks_keys),
    #                           key_hash=key_hash, exp_time=86400)
    deleted = cache_session.delete_from_cache(cache_key)
    print (f"Evict JWKs key result: {deleted} with key_hash = {key_hash}")

@log_errors("Add auth token to cache")
def add_auth_token_to_cache(query_id: str, env: str):
    query_key_hash = f"AUTH_TOKEN_{query_id}_{env}"

    cache_session = settings.get_cache_session()
    override_env = "prd"
    cache_session.update_expire_cache(query_id,
                                      json.dumps({
                                                  'token': AuthTokenLoader(override_env).token,
                                                  }),
                                      600,
                                      query_key_hash)



@log_errors("Add user metadata to cache")
async def add_user_metadata_to_cache(user_id: str, env: str, system: str, extra_context: str = None) -> None:
    def _context_string_to_dict(s):
        try:
            # Split the string by commas to get the key-value pairs
            pairs = s.split(',')

            # Initialize an empty dictionary
            result_dict = {}

            # Process each key-value pair
            for pair in pairs:
                # Split each pair by the colon to get the key and value
                key_value = pair.split(':', 1)  # Split only at the first colon
                if len(key_value) == 2:  # Ensure it's a valid key-value pair
                    key = key_value[0].strip()  # Strip any leading/trailing spaces
                    value = key_value[1].strip()  # Strip any leading/trailing spaces
                    result_dict[key] = value  # Add to the dictionary
        except Exception as e:
            return {}
        return result_dict
    if system.lower().startswith('nvinfo'):
        source = 'nvinfo-orchestrator'
    elif system.lower().startswith('nvbot') or system.lower().startswith('nvhelp'):
        source = 'nvhelp'
    else:
        source = system.lower()
    query_key_hash = f"user_metadata:{env.lower()}{source}"
    # print ("query_key_hash: ", query_key_hash)
    location = _context_string_to_dict(extra_context).get("Location") or "United States"

    cache_session = settings.get_cache_session()
    person_data = await get_person_data(NT_ACCOUNT_ID)

    if person_data is None:
        print(f"Failed to get person data from {source}")
        logging.error(f"Failed to get person data from {source}")
        return
    metadata_response = person_data.copy().get("response")
    metadata_response["location"] = location
    metadata_response["country"] = location
    metadata_response["id"] = user_id
    metadata = {"successful": True, "timeStamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
                "response": metadata_response}
    # print (metadata)
    cache_session.update_expire_cache(NT_ACCOUNT_ID,
                                      json.dumps(metadata),
                                      24 * 3600,
                                      query_key_hash)

    if source.lower() == "nvinfo-orchestrator":
        for expert_source in ["nvinfo_holiday_expert", "nvinfo_expert_sharepoint", "nvhelp"]:
            query_key_hash = f"user_metadata:{env.lower()}{expert_source}"
            cache_session.update_expire_cache(NT_ACCOUNT_ID,
                                              json.dumps(metadata),
                                              24 * 3600,
                                              query_key_hash)


def get_api_token(query_id: str, env: str) -> dict:
    query_hash_key = f"AUTH_TOKEN_{query_id}_{env}"
    token_info = get_cache_session().get_from_cache(query_id, query_hash_key)
    user_token = json.loads(token_info) if token_info else {}
    return user_token


@log_errors("Get header from cache")
def get_header(query_id: str, env: str) -> dict:
    user_token = get_api_token(query_id, env)
    override_env = "prd"
    user_token = {
        "token": AuthTokenLoader(override_env).token
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "device-initiated": "true" if user_token.get("device_initiated") else "false",
        "Authorization": f'Bearer {user_token.get("token")}',
        "query_id": query_id,
    }
    return headers


@log_errors("Create header")
def create_header(auth_token: str, **kwargs):
    # user_token = get_api_token(query_id)
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer {auth_token}",
        **kwargs
    }
    return headers


def make_request(
        method: RequestType,
        url: Optional[str] = None,
        sub_endpoint: Optional[str] = None,
        decode_json: bool = True,
        **kwargs,
) -> Dict[str, Any]:
    """
    Makes a request to the specified endpoint using the given HTTP method.

    Args:
        method (str): The HTTP method to use for the request ('GET' or 'POST').
        endpoint (str): The endpoint to send the request to.
        **kwargs: Additional keyword arguments to pass to the underlying request library.

    Returns:
        dict: The JSON response from the server.
    """

    # Get the full URL for the request, or the health check URL if no endpoint is provided.
    if sub_endpoint:
        url = f"{url}/{sub_endpoint}"

    if method == RequestType.GET:
        response = requests.get(url, **kwargs)
    elif method == RequestType.POST:
        response = requests.post(url, **kwargs)
    elif method == RequestType.DELETE:
        response = requests.delete(url, **kwargs)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")

    # Check for HTTP error codes
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        print(f"Error processing the NeMo request to '{url}':\n{e.response.text}")
        raise e

    return response.json() if decode_json else response
