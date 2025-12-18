import json
import logging
import time
from io import BytesIO
import os
from typing import Optional

import boto3
import requests
from fastapi import UploadFile, File
from starlette_context import context

from configs.settings import get_settings
from constants import common
from controllers.auth.auth_token_loader import AuthTokenLoader

from data_models.api.run_maker import RunMakerRequest, DatasetConfig, StorageType
from data_models.dataset_handler import DataContentOutputConfig
from nvbot_models.request_models.evaluation_request import EvaluationRunStatus, UserEvaluationProject, \
    UserEvaluationDataset
from nvbot_models.request_models.evaluation_request import UserEvaluationRunData
from nvbot_utilities.utils import api_handler
from service_library.constants import NT_ACCOUNT_ID, NT_ACCOUNT_NAME
from service_library.nemo_ms.nemo_service_helper import extract_nemo_eval_metadata
from service_library.utils.data_helper import byte_content_convertor, safe_json_loads
from service_library.utils.logging import log_errors
from service_library.utils.request_helpers import create_header

from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

MAX_RETRY_COUNT = 3
SLEEP_TIME = 2


def handle_connection_error(retry_count: int, url: str, error: str) -> int:
    if retry_count < 3:
        retry_count += 1
        time.sleep(SLEEP_TIME)
    else:
        logger.error('Exceeded max retries. Error getting response from api %s: %s', url, error)

    return retry_count


class DatabaseHandler:

    def __init__(self, config: dict = None):
        # env = "dev"

        self.config = config if config else {"env": "dev"}
        env = "prd" #config.get("env", "dev")
        self.auth_token = AuthTokenLoader(env).token
        self.service_url = get_settings().NVBOT_SERVICES_URL

    @log_errors("Insert Evaluation History")
    def add_to_evaluation_history(self, run_request: RunMakerRequest, dataset_metadata, evaluation_metadata,
                                  flowconfig_metadata, output_url: str = "",
                                  status: str = str(EvaluationRunStatus.STARTED.value)):
        output_json = safe_json_loads(output_url)
        print(f"+ add_to_evaluation_history output_json for project {run_request.Project}: ", output_json)
        print(f"+ trace_id for project {run_request.Project}: ", context.get("trace_id", "none"))
        # debug
        if output_json is None:
            print("output_json is None, from output_url", output_url)
        data = UserEvaluationRunData(
            Id=0,
            NtAccount=run_request.UserId or NT_ACCOUNT_ID,
            Username=run_request.UserName or NT_ACCOUNT_NAME,
            Project=run_request.Project,
            ProjectId=run_request.ProjectId,
            Status=status,
            Metadata=json.dumps(extract_nemo_eval_metadata(run_request)),
            RunType=run_request.RunType,
            OutputUrl=output_url,
            DatasetMetadataJson=dataset_metadata,
            EvaluationMetadataJson=evaluation_metadata,
            FlowConfigMetadataJson=flowconfig_metadata,
            Tag1=run_request.Parameters.get("tag1") if run_request.Parameters else None,
            Tag2=run_request.Parameters.get("tag2") if run_request.Parameters else None,
            NemoEvalId=output_json.get("id") if output_json else "",
            EvalType=output_json.get("eval_type") if output_json else "",
            Trace=context.get("trace_id", "") or ""
        )

        print(f'NemoEvalId = {output_json.get("id") if output_json else ""}')
        print(f'EvalType = {output_json.get("eval_type") if output_json else ""}')

        headers = create_header(self.auth_token)
        url = f"{self.service_url}/evaluation/history"
        print("Evaluation History attributes:", json.dumps(data.dict()))
        response = api_handler.post_data(url=url, headers=headers, request_body=json.dumps(data.dict()))
        print(f"Add Evaluation History response: {response}")
        if response['status']:
            logger.info(f"Saved evaluation history for {output_url}")
            return response.get("details")
        raise Exception("Failed to save evaluation history")

    @log_errors("Get Evaluation History")
    def get_evaluation_history(self, history_id: int, params=None):
        url = f"{self.service_url}/evaluation/history/{history_id}"
        headers = create_header(self.auth_token)
        response = api_handler.get_data(url=url, headers=headers)

        logger.info(f"Retrieved evaluation history id: {history_id}")
        return response

    @log_errors("Get Evaluation History Details")
    def get_evaluation_history_details(self, history_id: int, params=None):
        url = f"{self.service_url}/evaluation/historydetails/{history_id}"
        headers = create_header(self.auth_token)
        response = api_handler.get_data(url=url, headers=headers)

        logger.info(f"Retrieved evaluation history detail by history_id: {history_id}")

        url = f"{self.service_url}/evaluation/history/{history_id}"
        history_response = api_handler.get_data(url=url, headers=headers)
        if response and len(response) > 0:
            response[0]["history"] = history_response
        return response

    @log_errors("Delete Evaluation History")
    def delete_evaluation_history(self, history_id: int, params=None):
        url = f"{self.service_url}/evaluation/history/{history_id}"
        headers = create_header(self.auth_token)
        retry_count = 0
        while retry_count < 5:
            try:
                api_response = requests.delete(
                    url,
                    headers=headers,  # Optional headers (like auth tokens)
                )
                if api_response.ok:
                    logging.info(f"Deleted evaluation history id: {history_id}")
                    return api_response.json()
                elif str(api_response.status_code).startswith('5'):
                    retry_count = handle_connection_error(retry_count, url, api_response.text)
                else:
                    logger.warning(
                        f'Error HTTP Status code: {api_response.status_code}\nmessage: \n\t{api_response.text}')
                    raise Exception(api_response.text)
            except ConnectionError as ex:
                retry_count = handle_connection_error(retry_count, url, str(ex))
            except Exception as ex:
                logger.error('Error getting response from api %s: %s', url, ex)
                break

    @log_errors("Get Evaluation History List")
    def get_evaluation_history_records(self, filters: dict, pagination: dict = None):
        url = f"{self.service_url}/evaluation/history"
        headers = create_header(self.auth_token)

        # NEED FIX
        if pagination:
            filters.update(pagination)
        response = api_handler.get_data(url=url, headers=headers, params=filters)
        # return object:
        #  "total": ,
        #  "page": ,
        #  "size": ,
        #  "pages": ,
        #  "items": []
        return response if response else None

    @log_errors("Update Evaluation History Status")
    def update_evaluation_history_and_details(self, run_data: UserEvaluationRunData):
        assert run_data.Id is not None, f"Update history requires id field to be not None, get {run_data.Id}"

        url = f"{self.service_url}/evaluation/history?history_id={run_data.Id}"
        headers = create_header(self.auth_token)

        data = run_data.dict()
        # print(f"Update Evaluation History and Detail request: {json.dumps(data)}")
        update_response = api_handler.put_data(
            url=url,
            headers=headers,
            request_body=json.dumps(data)
        )
        print(f"Update Evaluation History and Detail response: {update_response}")
        assert update_response, "Failed to update Evaluation History and Details"
        logger.info(
            f"Updated evaluation history: {update_response.get('id')} for project: {update_response.get('project')}")
        print(f"Updated evaluation history: {update_response.get('id')} ")

        return update_response

    @log_errors("Get Evaluation Project")
    def get_evaluation_project(self, filters: dict):
        url = f"{self.service_url}/evaluation/project"
        headers = create_header(self.auth_token)
        response = api_handler.get_data(url=url, headers=headers, params=filters)

        logging.info(f"Retrieved evaluation projects")
        return response.get("items") if response else None

    @log_errors("Insert Evaluation Project")
    def add_evaluation_project(self, project_data: UserEvaluationProject):
        headers = create_header(self.auth_token)
        url = f"{self.service_url}/evaluation/project"

        response = api_handler.post_data(url=url, headers=headers, request_body=json.dumps(project_data.dict()))
        if response['status']:
            logger.info(f"Saved evaluation project with id: {response.get('id')}")
            return response.get("details")
        raise Exception("Failed to save evaluation project")

    @log_errors("Insert Evaluation Project")
    def add_evaluation_dataset(self, dataset_data: UserEvaluationDataset):
        headers = create_header(self.auth_token)
        url = f"{self.service_url}/evaluation/project/dataset"

        response = api_handler.post_data(url=url, headers=headers, request_body=json.dumps(dataset_data.dict()))
        if response['status']:
            logger.info(f"Saved evaluation project with id: {response.get('id')}")
            return response.get("details")
        raise Exception("Failed to save evaluation project")

    @log_errors("Get Evaluation History Details Result as DatasetConfig")
    def get_detail_result_as_datasetconfig(self, history_id: str) -> Optional[DatasetConfig]:
        # check details availability
        history_details = DatabaseHandler({'env': 'dev'}).get_evaluation_history_details(history_id=history_id)
        if not history_details:
            return None
        # example: {"url": "https://devbot-api.nvidia.com/evaluation/datasets/download/dataset-DAWLcL6iT52iApyYJLRroW/file/contents/{filepath}?file_path=nvinfo_mixtral_agent_complete_evaluation-sandbox-2024-11-02_23:59:07_PDT-correctnessanswer4.07-helpfulness4.41.xlsx"}
        parsed_url = urlparse(history_details[0]['results_metadata']['url'])

        # Find the "download" part in the path, then extract the dataset id
        path_parts = parsed_url.path.split('/')
        if "download" in path_parts:
            download_index = path_parts.index("download")
            dataset_id = path_parts[download_index + 1]
        else:
            dataset_id = None  # Handle case if "download" is not in the path

        # Extract the file name from query parameters
        query_params = parse_qs(parsed_url.query)
        file_name = query_params.get('file_path', [None])[0]
        if dataset_id and file_name:
            return DatasetConfig(
                Engine=StorageType.DATASTORE.value,
                DatasetId=dataset_id,
                Files=[
                    file_name
                ],
                HistoryId=history_id,
            )
        return None
