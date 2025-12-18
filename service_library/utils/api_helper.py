import json
import logging
import time
from typing import Any

import requests


log = logging.getLogger('APIHelper')


MAX_RETRY_COUNT = 10
SLEEP_TIME = 10


def handle_connection_error(retry_count: int, url: str, error):
    if retry_count < MAX_RETRY_COUNT:
        retry_count += 1
        log.warning('Got connection error: %s. Retrying in %s seconds', error, SLEEP_TIME)
        time.sleep(SLEEP_TIME * retry_count)
    else:
        log.error(f'Exceeded max retries = {MAX_RETRY_COUNT}. Error getting response from api %s: %s', url, error)

    return retry_count


def get_data(url: str, headers: dict, params=None, timeout=None, body=None) -> tuple[Any, bool]:
    retry_count = 0
    status = False
    while retry_count < MAX_RETRY_COUNT:
        retry_count += 1
        try:
            api_response = requests.get(url, headers=headers, params=params, timeout=timeout, data=json.dumps(body))
            if api_response.ok:
                return api_response, True
            elif api_response.status_code == 404:
                return api_response, True
            else:
                print(f"🙃 ERROR in GET request to {url}:", api_response.text)
        except ConnectionError as ex:
            retry_count = handle_connection_error(retry_count, url, ex)
        except Exception as ex:
            log.error('🙃 Error in GET response from api %s: %s', url, ex)
            return api_response, False

    return api_response, status



def post_json(url: str, headers: dict, request_body: dict, auth=None, timeout=None) -> tuple[
    Any, bool, int, int, float]:
    retry_count, avg_duration, time_accumulator = 0, 0,  0.0

    api_response, status = None, False
    while retry_count < MAX_RETRY_COUNT:
        try:
            start_time = time.time()

            if auth:
                api_response = requests.post(url, json=request_body, headers=headers, auth=auth, timeout=timeout)
            else:
                api_response = requests.post(url, headers=headers, json=request_body, timeout=timeout)

            elapsed_time = time.time() - start_time
            time_accumulator += elapsed_time

            if api_response.ok or api_response.status_code == 404:
                avg_duration = time_accumulator / (retry_count + 1)
                return api_response, True, api_response.status_code, retry_count, round(avg_duration,2)
            elif api_response.status_code in [504, 503]:
                retry_count += 1
                retry_count = handle_connection_error(retry_count, url, api_response.text)
            else:
                print(f"🙃ERROR in post request to {url}:", api_response.text)
        except requests.ConnectionError as ex:
            retry_count = handle_connection_error(retry_count, url, ex)
        except Exception as ex:
            log.error('🙃Error in POST response from api %s: %s', url, ex)
            avg_duration = time_accumulator / max(1, retry_count)  # Avoid division by zero
            return api_response, False, getattr(api_response, 'status_code', 500), retry_count, round(avg_duration,2)
        retry_count += 1
    avg_duration = time_accumulator / max(1, retry_count)  # Ensure valid division
    return api_response, status, 500, retry_count, round(avg_duration,2)


# def post_json(url: str, headers: dict, request_body: dict, auth = None, timeout=None) -> tuple[Any, bool, int, int]:
#     retry_count = 0
#     timeout_accumulator = 0
#
#     api_response, status = None, False
#     while retry_count < MAX_RETRY_COUNT:
#         try:
#             if auth:
#                 api_response = requests.post(url, json=request_body, headers=headers, auth=auth)
#             else:
#                 api_response = requests.post(url, headers=headers, json=request_body, timeout=timeout)
#             # print ("api_response:", api_response)
#             if api_response.ok:
#                 return api_response, True, api_response.status_code, retry_count
#             elif api_response.status_code == 404:
#                 return api_response, True, api_response.status_code, retry_count
#             elif api_response.status_code in [504, 503]:
#                 retry_count += 1
#                 retry_count = handle_connection_error(retry_count, url, api_response.text)
#             else:
#                 print( f"🙃 ERROR in post request to {url}:" , api_response.text)
#         except ConnectionError as ex:
#             retry_count = handle_connection_error(retry_count, url, ex)
#         except Exception as ex:
#             log.error('🙃Error in POST response from api %s: %s', url, ex)
#             return api_response, False, api_response.status_code, retry_count
#
#     return api_response, status, 500, retry_count
#

def post_data(url: str, headers: dict, request_body: str, timeout=None) -> tuple[dict, bool]:
    retry_count = 0
    api_response, status = None, False
    while retry_count < MAX_RETRY_COUNT:
        retry_count += 1
        try:
            api_response = requests.post(url, headers=headers, data=request_body, timeout=timeout)
            if api_response.ok:
                return api_response.json(), True
            elif api_response.status_code == 404:
                return api_response, True
            elif api_response.status_code in [504, 503]:
                retry_count = handle_connection_error(retry_count, url, api_response.text)
            else:
                print( f"🙃 ERROR in request to {url}:" , api_response.text)
        except ConnectionError as ex:
            retry_count = handle_connection_error(retry_count, url, ex)
        except Exception as ex:
            log.error('🙃 Error in POST response from api %s: %s', url, ex)
            return api_response, False

    return api_response, status

def put_data(url: str, headers: dict, request_body: str, timeout=None) -> tuple[Any, bool]:
    retry_count = 0
    api_response, status = None, False
    while retry_count < MAX_RETRY_COUNT:
        retry_count += 1
        try:
            api_response = requests.put(url, headers=headers, data=request_body, timeout=timeout)
            if api_response.ok:
                return api_response, True
            elif api_response.status_code == 404:
                return api_response, True
            elif api_response.status_code == 409:
                return {"api_response_code": 409}, True
            else:
                print(f"🙃 ERROR in PUT request to {url}:", api_response.text)
        except ConnectionError as ex:
            retry_count = handle_connection_error(retry_count, url, ex)
        except Exception as ex:
            log.error(' 🙃 Error in PUT response from api %s: %s', url, ex)
            return api_response, False

    return api_response, status
