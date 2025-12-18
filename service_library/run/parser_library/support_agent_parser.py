import json
import re
from typing import Any

from service_library.run.parser_library.common_parser import parse_response_with_links
from service_library.utils.data_helper import safe_json_loads
from service_library.utils.logging import log_errors

import json
import math
from typing import Any

from service_library.nemo_ms.nemo_service_helper import convert_number
from service_library.utils.logging import log_errors


@log_errors("generate request")
def generate_request(row):
    """
        Convert a row with 'Query' and 'Channel ID' into request payload
        """
    return {
        "input_message": row["Query"] or "",
        "channel_ids": [row["Channel"]],
        "exclude_thread_url": row["Thread"] or "",
    }


@log_errors("Parse response")
def parse_response(data):
    value_response, links, response_text = {}, [], ""
    try:
        if data:
            parsed_response = parse_response_with_links(data.get("value"), "response")

            response_text = parsed_response.get("Response", "")
            links = parsed_response.get("Links", [])
    except Exception as err:
        print(f"Cant process {data}: {err}")

    return {
        "Response": response_text,
        "Links": links,
        "Confidence": value_response.get("confidence"),
        "Sources": value_response.get("sources"),
    }
