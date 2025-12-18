import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import Optional, List, Any, Tuple

import requests
from langchain.agents.chat.output_parser import ChatOutputParser
from langchain.agents.output_parsers import JSONAgentOutputParser

from configs.settings import get_settings
from nvbot_models.request_models.bot_maker_request import BotmakerRequestParameter
from nvbot_models.request_models.fulfillment_request import FulfillmentRequest
from service_library.constants import NT_ACCOUNT_ID
from service_library.url_wrappers.url_wrapper import URLWrapper
from service_library.utils.api_helper import post_json
from service_library.utils.logging import log_errors
from service_library.utils.request_helpers import create_header, add_auth_token_to_cache, get_header, \
    add_user_metadata_to_cache

# Regular expressions
tool_pattern = r"tool='(.*?)'"
tool_input_pattern = r"tool_input='(.*?)'"
action_pattern = r'Action:\s*([\w]+)'
action_input_pattern = r'Action Input:\s*(\[.*?\])'


async def fetch_post_chatbot_request_response(fulfillment_request: FulfillmentRequest, chatbot_url: str, env: str,
                                              extra_context: dict = None) -> Tuple[dict, int, int | float]:
    response_text = None
    response = {}
    retry_count, _duration_avg = 0, 0

    nvbot_platform_config = fulfillment_request.PlatformConfig
    system = fulfillment_request.System

    # DEBUG ONLY
    # fulfillment_request.Query = ""
    chatbot_request_dict = fulfillment_request.dict()

    add_auth_token_to_cache(fulfillment_request.QueryId, env)
    await add_user_metadata_to_cache(fulfillment_request.UserId, env, fulfillment_request.System, extra_context)

    headers = get_header(fulfillment_request.QueryId, env)
    is_stream = fulfillment_request.Parameters.IsStream
    use_graph_chatbot_url = ('GraphConfig' in list(nvbot_platform_config.FlowConfig.dict().keys())) and (
            nvbot_platform_config.FlowConfig.GraphConfig is not None)
    print("chatbot_request_dict: ", chatbot_request_dict)

    if use_graph_chatbot_url:
        # Note: temp solution for graph orchestrator
        api_response, status, status_code, retry_count, duration_avg = post_json(chatbot_url, headers, chatbot_request_dict)
        if api_response is not None:
            # print(f"Request Header Auth: {headers.get('Authorization')}")
            # print(f"Request SessionId: {chatbot_request_dict.get('SessionId')}")
            # print(f"Request QueryId: {chatbot_request_dict.get('QueryId')}")
            # print(f"Response status code: {api_response.status_code}")
            logging.info(f"Response status code: {api_response.status_code}")
            if api_response.ok:
                is_stream = fulfillment_request.Parameters.IsStream
                if is_stream:
                    response_text = api_response.text
                    response_list = response_text
                else:
                    json_response = api_response.json()
                    if json_response:
                        response_list = json_response.get("response", [])

                        final_state = list(response_list[-1].values())[0] if response_list else None
                        assert final_state is not None, f"Failed to get agent final state"
                        response_text = final_state.get("messages", [])[-1].get("content")

                response = {"Status": api_response.status_code, "Response": response_list, "Text": response_text}
            else:
                print(f"🚨Error Status from {chatbot_url}: {api_response.status_code}")
                response = {"Status": api_response.status_code, "Response": {"Text": api_response.text}}
    else:
        if system.lower() == "scout":  # + trip report
            chatbot_request_dict_parameters = fulfillment_request.Parameters.dict()  # chatbot_request_dict.get("Parameters", {})
            chatbot_request_dict_request_parameters = BotmakerRequestParameter(
                NvidiaID=NT_ACCOUNT_ID,
                Source=system.lower(),
                SourceType="private",
                Tz="America/Los_Angeles",
                TzLabel="Pacific Daylight Time",
                TzOffset=-25200,
                IsStream=False
            ).dict()
            # chatbot_request_dict_parameters["RequestParameters"] = chatbot_request_dict_request_parameters
            chatbot_request_dict["Parameters"] = {"RequestParameters": chatbot_request_dict_request_parameters}

        api_response, status, status_code, retry_count, duration_avg = post_json(chatbot_url, headers, chatbot_request_dict)
        if api_response:
            # print(f"Response status code: {api_response.status_code}")
            logging.info(f"Response status code: {api_response.status_code}")
            if api_response.ok:
                if is_stream:
                    response = {"Response": api_response.text}
                    response_text = api_response.text
                else:
                    json_response = api_response.json()
                    if isinstance(json_response, str):
                        response = {"Response": json_response, "Text": json_response}
                    else:
                        response = json_response
                response = {"Status": api_response.status_code, **response}

            else:
                print(f"🚨Error Status from {chatbot_url}: {api_response.status_code}")
                response = {"Status": api_response.status_code, "Response": {"Text": api_response.text}}

    assert api_response is not None, f"⚠️Unable to retrieve response for {json.dumps(chatbot_request_dict)}"

    # fail safe
    if response_text is None and isinstance(response, dict):
        # print(f'Received response: {response.get("Response")}')
        # logging.info(f'Received response: {response.get("Response")}')
        response_text = response.get("Response", {}).get("Text") if response and response.get(
            "Response") else None
        if not response_text:
            response_text = str(response.get("Response", {}).get("Json", {}).get("Text", {}).get("text"))
    return {
        "status": int(response.get("Status")) or api_response.status_code,
        "response": response.get("Response", {}),
        "response_text": response_text,
    }, retry_count, duration_avg


## Langchain Response

def extract_agent_action(input_str: str, key: str):
    """
    Extract the agent action and action input from the agent action string

    @param agent_action_str: agent action string
    """
    key = key.lower()

    if key == "tool":
        tool_match = re.search(tool_pattern, input_str)
        if tool_match:
            tool = tool_match.group(1)
        else:
            tool = ""
        return tool

    elif key == "tool_input":
        # Search for tool_input
        tool_input_match = re.search(tool_input_pattern, input_str)
        if tool_input_match:
            tool_input = eval(tool_input_match.group(1))
        else:
            tool_input = ""
        return tool_input

    elif key == "action":
        # Search for Action
        action_match = re.search(action_pattern, input_str)
        if action_match:
            action = action_match.group(1)
        else:
            action = ""
        return action

    elif key == "action_input":
        # Search for Action Input
        action_input_match = re.search(action_input_pattern, input_str, re.DOTALL)
        if action_input_match:
            action_input = action_input_match.group(1)
        else:
            action_input = ""
        return action_input

    return ""


def extract_tool(agent_action_str):
    """
    Extract the agent action and action input from the agent action string

    @param agent_action_str: agent action string
    """

    return extract_agent_action(agent_action_str, "action")


def extract_tool_input(agent_action_str):
    """
    Extract the agent action and action input from the agent action string

    @param agent_action_str: agent action string
    """
    # print(f"{agent_action_str}")
    # tool_input = ""
    # regex_pattern = r"Action: (.*?)[\n]*Action Input: (.*)"
    # match = re.search(regex_pattern, agent_action_str)
    # if match:
    #     _action = match.group(1)
    #     tool_input = match.group(2)
    return extract_agent_action(agent_action_str, "action_input")


def get_by_index_key(obj_data: dict | list, key: int | str):
    if isinstance(obj_data, dict):
        return obj_data.get(key, None)
    elif isinstance(obj_data, list):
        if isinstance(key, int) and 0 <= key < len(obj_data):
            return obj_data[key]
        else:
            return None
    else:
        raise TypeError("The data must be a list or a dictionary")


NVBOT_CITATION_HEADER = "For more details, please refer to the following links:"
SCOUT_CITATION_HEADER = "Here are the sources used to generate this response:"


def response_citation_parser(json_response):
    if isinstance(json_response, dict):
        if json_response.get("Citation"):
            return json_response
        response_text = json_response.get("Response")
    elif isinstance(json_response, str):
        response_text = json_response
        json_response = {}
    else:
        print(f"Failed to parse response citation, given: {json_response}")
        return {}
    print("📝Response Text:\n", response_text)
    before, keyword_used, after = response_text.partition(SCOUT_CITATION_HEADER)
    if not keyword_used:
        before, keyword_used, after = response_text.partition(NVBOT_CITATION_HEADER)

    json_response["LLM response"] = before.strip()
    json_response["Citation"] = after.strip()
    return json_response


def expert_streaming_output_parser(text, expert_name=""):
    text = URLWrapper().remove_embed_inline_images(text)
    strs = text.split("status_update_done")

    if len(strs) > 1:
        return strs[1]

    return text


def response_suggested_question_remover(json_response):
    response_text = ""
    if isinstance(json_response, dict):
        if json_response.get("Citation"):
            return json_response
        response_text = json_response.get("Response")
    elif isinstance(json_response, str):
        response_text = json_response
        json_response = {}

    return re.sub(r'<picker>.*?</picker>', '', response_text, flags=re.DOTALL)

def response_suggested_question_parser(json_response):
    response_text = ""
    if isinstance(json_response, dict):
        if json_response.get("Citation"):
            return json_response
        response_text = json_response.get("Response")
    elif isinstance(json_response, str):
        response_text = json_response
        json_response = {}
    else:
        match = re.search(r'<picker>(.*?)</picker>', response_text, re.DOTALL)
        if not match:
            return None  # Return an empty list if no <picker> tags found

    match = re.search(r'<picker>(.*?)</picker>', response_text, re.DOTALL)
    if not match:
        return None  # Return an empty list if JSON parsing fails

    picker_content = match.group(1)

    # Step 2: Load the extracted content as JSON
    try:
        picker_json = json.loads(picker_content)
    except json.JSONDecodeError:
        return picker_content
    # Step 3: Extract all "Label" values in a list
    labels = [option.get("Label").strip() for option in picker_json.get("Options", [])]

    return labels


## Orchestrator Response

def parse_string_to_dict(text):
    result = text
    try:
        result = json.loads(text)
    except json.decoder.JSONDecodeError:
        result = ChatOutputParser().parse(text).dict()
    except Exception as e:
        print(f"Error processing data `{text}`: {e}")
        logging.info(f"Error processing data `{text}`: {e}")
    finally:
        return result


@log_errors("Get Orchestrator response")
def get_orchestrator_response(response_list):
    if not isinstance(response_list, list):
        return response_list
    elif len(response_list) == 0:
        return response_list
    final_state = list(response_list[-1].values())[0] if response_list else None
    assert final_state is not None, f"Failed to get agent final state"
    response_text = final_state.get("messages", [])[-1].get("content")
    return expert_streaming_output_parser(response_text)


@log_errors("Get NVInfo Orchestrator streaming answer response")
def get_nvinfo_orchestrator_streaming_answer_response(response_list):
    if not isinstance(response_list, list):
        return response_list
    elif len(response_list) == 0:
        return response_list
    final_state = list(response_list[-1].values())[0] if response_list else None
    assert final_state is not None, f"Failed to get agent final state"
    messages = final_state.get("messages", [])

    messages_contents = []
    last_stream_answer_index = -1  # Initialize to -1 to indicate "not found"

    # Loop through the messages to find the last occurrence of "stream_answer"
    for i, element in enumerate(messages):
        if element.get("content") == "stream_answer":
            last_stream_answer_index = i  # Keep updating with the latest index

    if last_stream_answer_index != -1:
        for element in messages[last_stream_answer_index + 1:]:
            content = element.get("content")
            if content:
                messages_contents.append(content)

    return_results = {}

    if len(messages_contents) == 1:
        response = expert_streaming_output_parser(messages_contents[0])
        return_results["Response"] = response
        return_results.update(response_citation_parser(response))

    elif len(messages_contents) > 1:
        response = expert_streaming_output_parser(messages_contents[0])
        return_results["Response"] = response
        return_results.update(response_citation_parser(response))
        suggested_question = response_suggested_question_parser(' '.join(messages_contents[1:]))
        print("📝Suggested Question:\n", suggested_question)
        return_results["Suggested Question"] = response_suggested_question_parser(' '.join(messages_contents[1:]))

    else:
        response = get_orchestrator_response(response_list)
        return_results["Response"] = response
        return_results.update(response_citation_parser(response))
    # print("📝Streaming response:\n", response)
    return return_results


@log_errors("Get Orchestrator response 2")
def get_orchestrator_response2(response_list: Optional[List[str]], keys: Optional[List[str]] = None):
    if not isinstance(response_list, list):
        return response_list
    elif len(response_list) == 0:
        return response_list
    final_state = list(response_list[-1].values())[0] if response_list else None
    assert final_state is not None, f"Failed to get agent final state"
    response_text = final_state.get("messages", [])[-1].get("content")

    return expert_streaming_output_parser(response_text)


def filter_dict(input_dict, keys):
    return {key: input_dict[key] for key in keys if key in input_dict}


def process_orchestrator_node_data(json_response, node_keys_list=["agent_scratchpad"],
                                   response_type_excludes=["human"]):
    if json_response:
        results = {}
        if True:
            for node, node_metadata in json_response.items():
                # print(f"\nNode: {node}")
                info = []
                if node in node_keys_list:
                    contents = defaultdict(set)
                    for obj in node_metadata:
                        name = obj.get("name", None)
                        type = obj.get("type", "")
                        if response_type_excludes and type not in response_type_excludes:
                            key = "_".join([name, type]) if name else type
                            content = obj["content"]

                            contents[key].add(content)

                    # print (contents)
                    if contents:
                        parsed_contents = {}
                        for key, vals in contents.items():
                            parsed_contents[key] = [parse_string_to_dict(val) for val in vals]
                        results[node] = parsed_contents
        # print ("results", results)
        return results


@log_errors("Response parsing for stringfy json")
def decompose_stringify_json_data(stringify_json_response: str, json_key: str = None):
    json_response = stringify_json_response
    try:
        if isinstance(stringify_json_response, str):
            json_response = json.loads(stringify_json_response)
        elif isinstance(stringify_json_response, dict):
            json_response = stringify_json_response
        else:
            raise ValueError("Expected input to be a string or dictionary")

    except Exception as ex:
        print(f"[warn] Failed to loads input string to json: {stringify_json_response}")
        logging.warning(f"Failed to loads input string to json: {stringify_json_response}")

    if json_key and isinstance(json_response, dict):
        return json_response.get(json_key, "")
    else:
        return json_response


@log_errors('Json Decoder')
def json_decoder(obj: Any):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj.__dict__


@log_errors('Json wrapper removal')
def json_wrapper_removal(input_string: str):
    try:
        input_string = re.sub(r'^```json\s*|\s*```$', '', input_string.strip(), flags=re.DOTALL)
        return json.loads(input_string)
    except Exception as ex:
        print(f"[warn] Failed to remove json wrapper: {input_string}")
        logging.warning(f"Failed to loads input string to json: {input_string}")

    return input_string


@log_errors('Object to string')
def object_to_string(data: Any):
    try:
        json_data = json.dumps(data, default=lambda o: json_decoder(o), indent=4)
    except Exception as ex:
        json_data = json.dumps(data, default=str)
    return json_data

# load_as_json = lambda json_string: json.loads(json_string)

@log_errors('Validate URL')
def validate_urls(cls, markdown_text: str):
    unique_urls = URLWrapper.extract_links(markdown_text)
    errors = []

    for url in unique_urls:
        try:
            # Send a GET request to check the URL status code
            response = requests.get(url, timeout=5)
            if "sharepoint.com" not in url and response.status_code != 200:
                errors.append(
                        f"{url} | Error status code: {response.status_code}, information: {response.text}") if response.text else errors.append(
                        f"{url} | Error status code: {response.status_code}")
        except requests.RequestException as e:
            # Append errors for failed requests (e.g., timeout or connection error)
            errors.append(f"{url} | Error: {str(e)}")

    return errors if errors else None


@log_errors('Extract refresh response')
def extract_refresh_response(raw: str, marker: Optional[str] = None) -> str:
    """
    Return the last segment of the raw response after splitting on the refresh marker.
    Marker defaults to env var EVAL_REFRESH_SPLIT_MARKER or literal "refresh_response_start".
    """
    if not isinstance(raw, str) or not raw:
        return raw
    marker_value = "start_refresh_response:"
    try:
        parts = raw.split(marker_value)
        return parts[-1].strip()
    except Exception:
        return raw.strip()