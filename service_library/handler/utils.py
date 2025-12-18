import json
import logging
from typing import Optional

import aiohttp
import requests
import tiktoken

# from nvbot_utilities.utils.api_handler import get_data
from configs.settings import get_settings

logger = logging.getLogger("Utils")

MULESOFT_HEADER = {
    "client_id": "",
    "client_secret": "",
    "Content-Type": "",
}


def create_header_payload(client_id: str, client_secret: str, content_type="application/json"):
    headers = MULESOFT_HEADER
    headers["client_id"] = client_id
    headers["client_secret"] = client_secret
    headers["Content-Type"] = content_type
    return headers


async def get_person_data(username: str) -> Optional[dict]:
    try:
        settings = get_settings()
        headers = create_header_payload(
            settings.MULESOFT_CLIENT_ID,
            settings.MULESOFT_CLIENT_SECRET,
        )
        url = settings.SINGLE_USER_URL.format(username)
        # response = get_data(url=url, headers=headers)
        async with aiohttp.ClientSession() as session:
            async with await session.get(url=url, headers=headers) as response:
                if response.status != 200:
                    error_source = "MULESOFT"
                    if hasattr(response, "source"):
                        logger.error(
                            "",
                            f"RestAPIError - Glean process_query error: \
                                                    status code : {response.status}, source : {error_source}",
                        )
                else:
                    return await response.json()
    except Exception as e:
        logger.error(f"error fetching user context from mulesoft api {e}")


def trim_text_to_n_tokens_openai(text: str, n_tokens: int, model_name: str):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    tokens = tokens[:n_tokens]
    trimmed_text = encoding.decode(tokens)
    return trimmed_text


def trim_text_to_n_tokens_nemo(text: str, n_tokens: int):
    # Sean's token hack for llama - TODO replace with nemo encoding / decoding model
    if n_tokens == 0:
        return text[:-100]
    else:
        return text[:int(n_tokens * 13 / 6)]
