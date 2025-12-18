import requests
import json
from urllib.parse import quote

from service_library.utils.logging import log_errors

headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "no-cache",
        "origin": "https://wwwstage.nvidia.com",
        "pragma": "no-cache",
        "priority": "u=1, i",
        "referer": "https://wwwstage.nvidia.com/",
        }

@log_errors("Fetch ir bot response")
def fetch_ir_bot_data(query):
    encoded_query = quote(query)
    url = f"https://api-stage.nvidia.com/services/irbot-app/userquery/{encoded_query}"

    response = requests.get(url, headers=headers)
    print ("fetch_ir_bot_data: ", response.status_code)
    # Check if the request was successful
    if response.status_code == 200:
        response_data = response.json()

        # Extract the 'data' key
        if 'data' in response_data:
            data = response_data['data']
            response_type = response_data['responseType']

            if response_type.lower() == "string":
                return f"{data}"
            caption = response_data.get('caption', "")
            columns = data.get('columns', [])
            values = data.get('values', [])
            # print ("columns", columns)
            # print ("values", values)
            response = [
                {columns[i]: value for i, value in enumerate(row)}
                for row in values
            ]
            # print ("ir bot response: \n", response)
            return f"{caption}\n\n{response}"
    else:
        return f"Request failed with status code {response.status_code}"
