import re
from service_library.utils.data_helper import safe_json_loads
from service_library.utils.logging import log_errors

@log_errors("Parse response with links")
def parse_response_with_links(value_response, response_key = None):
    links, response_text = [], ""
    try:
        if value_response:
            value_response = safe_json_loads(value_response, failback=value_response)
            if response_key:
                response_text = value_response.get(response_key, value_response)
            else:
                if isinstance(value_response, (dict, list, tuple, set)) and len(value_response) == 1:
                    response_text  = value_response[next(iter(value_response))]
                else:
                    response_text = str(value_response)

            # 2. Regex to extract URLs
            url_pattern = r'(https?://[^\s]+)'
            links = re.findall(url_pattern, response_text)
    except Exception as err:
        print(f"Cant process {value_response}: {err}")

    return {
        "Response": response_text,
        "Links": links,
    }