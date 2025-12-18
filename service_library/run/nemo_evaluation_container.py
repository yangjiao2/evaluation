from __future__ import annotations

import concurrent.futures
import inspect
import json
import math
import re

from configs.settings import get_settings
import logging
import os
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, cast, Callable
from uuid import UUID, uuid4

import aiohttp
import numpy as np
import pandas as pd
import pytz
import requests
import statistics
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers import EvaluatorCallbackHandler

from langsmith.evaluation import EvaluationResult
from pandas import DataFrame
from tqdm import tqdm
from configs.settings import get_settings, get_cache_session
from controllers.auth.auth_token_loader import AuthTokenLoader

from data_models.api.evaluation_result import EvalRunMetrics
from data_models.dataset_handler import DataContentOutputConfig
from data_models.api.run_maker import RunMakerRequest, NVBotEvaluationConfig, DatasetConfig, \
    EvaluationSchema, FlowConfigRequest, StorageType
from nvbot_models.request_models.evaluation_request import UserEvaluationRunData, EvaluationRunStatus, \
    UserEvaluationDetailData
# from nv_platform.nvbot_platform import NVBotPlatform
from nvbot_models.request_models.fulfillment_request import FulfillmentRequest
from nvbot_utilities.utils.datadog.custom_span_info import get_current_span, set_span_tags
from service_library.constants import CUSTOM_EVAL_INPUT_FILENAME, LOCAL_EVAL_RESULTS_TMP_FOLDER_WITH_END_SLASH, \
    NT_ACCOUNT_ID, NT_ACCOUNT_NAME, LOCAL_EVAL_RESULTS_TMP_FOLDER, LLM_AS_A_JUDGE_EVALUATOR_TYPE, \
    LLM_AS_A_JUDGE_FILE_NAME, CUSTOM_FILE_NAME, EVAL_STATUS_MAPPING, CUSTOM_EVALUATOR_TYPE, DATASET_FILENAME, \
    ANSWER_FILENAME, EVAL_ACCOUNT_ID
from service_library.handler.config_loader import ConfigLoader
from service_library.handler.database_handler import DatabaseHandler
from service_library.handler.datadog_handler import DatadogHandler
from service_library.handler.s3_dataset_handler import S3DatasetHandler
from service_library.nemo_ms.nemo_datastore import NeMoDataStore
from service_library.nemo_ms.nemo_evaluator import NeMoEvaluator
from service_library.nemo_ms.nemo_service_helper import prepare_llm_as_a_judge_nemoeval_dataset, \
    create_dataset_and_upload_folder, jsonl_writer, prepare_custom_nemoeval_dataset, \
    prepare_custom_nemoeval_payload, prepare_llm_as_a_judge_nemoeval_payload, \
    parse_custom_eval_aggregated_metrics, extract_numeric_value_from_llm_judgment, \
    extract_string_from_llm_judgment, convert_number
from service_library.notification.send_email import EmailServices
from service_library.run.comparison_container import ComparisonContainer
from service_library.run.parser_library.input_parser import request_parser
from service_library.run.parser_library.dict_parser import dict_parser
from service_library.run.run_container import RunContainer
from service_library.utils.configuration_helper import get_evaluation_config_by_nemo_evaluator_type
from service_library.utils.data_helper import json_writer, file_content_convertor, move_file, \
    get_file_extension, calculate_category_metrics, \
    flatten_dict_by_1_degree, convert_to_snake_case_separated_by_dot_notation
from service_library.utils.datasetconfig_helper import load_data_from_datasetconfig
from service_library.utils.logging import log_errors
from service_library.utils.pydantic_helper import is_valid_instance
from service_library.utils.request_helpers import create_header
from service_library.utils.run_helpers import generate_nemo_eval_local_folder_path, generate_short_uuid, \
    generate_unique_datastore_path, generate_excel_output, create_folder_if_not_exists, get_formatted_datetime

# from utility import generate_regression_filename

logger = logging.getLogger(__name__)


class NemoEvaluationRunContainer(RunContainer):
    """A container to help manage the state of an eval run."""

    input_mapper = None
    output_mapper = None

    def __init__(self, project: str, config: NVBotEvaluationConfig, env: str):
        super().__init__(project, config, env)

        self.input_mapper = None
        self.output_mapper = None

        self.metrics = {}
        self.temp_folderpath = None

        self.EVAL_URL = self.settings.NEMO_EVAL_URL
        self.DS_URL = self.settings.NEMO_DS_URL

        self.evaluators: List[NeMoEvaluator] = []
        self.datastorers: List[NeMoDataStore] = []
        self.name = f"nvbot-{project}"

    @log_errors('Trigger Nemo Eval')
    async def arun(
            self,
            request: RunMakerRequest,
            config: NVBotEvaluationConfig,
            **kwargs):

        results = []
        print('nemo run')
        for index, evaluator in enumerate(self.evaluators):
            evaluation_result = evaluator.launch()

            print("\nLaunched evaluation result:", evaluation_result)
            nemo_eval_id = evaluation_result.get('id')

            logger.info(f"Launched Evaluation: {nemo_eval_id}")
            use_nvcf_endpoint = self.EVAL_URL.endswith("7331")
            # request.Parameters = {}
            # request.Parameters["tag1"] = "cluster" if use_nvcf_endpoint else ""
            try:

                save_evaluation_history_response = _save_evaluation_history_to_database(request=request,
                                                                                       datastore_content=
                                                                                       self.datastorers[
                                                                                           index].datastore_content,
                                                                                       evaluation_result=evaluation_result)
                if save_evaluation_history_response is None:
                    print("Failed to insert evaluation history", evaluation_result)
                else:
                    print(f"Evaluation history saved with id: {save_evaluation_history_response.get('id')}")
            except Exception as ex:
                print(f"Exception in save evaluation history to database, {ex}")
                raise Exception(ex)
            evaluation_result.update({"eval_history_id": save_evaluation_history_response.get('id')})
            results.append(evaluation_result)

        return results

    @log_errors("Finish nemo eval run")
    def finish(self, request: RunMakerRequest, config: NVBotEvaluationConfig, run_results: dict):
        response = {"status": "success"}
        nemo_eval_id = run_results.get('id')
        results = {}
        try:
            allow_notification = config.EvaluationSchema.Notification
            if allow_notification:
                email_recipients = allow_notification.EmailRecipients
                print("allow_notification", allow_notification)
                if email_recipients:
                    logger.info(f"email recipients: {email_recipients}")
                    print("email_recipients", email_recipients)
                    if email_recipients:
                        email_service_response = EmailServices().send_email(
                            f"Launched eval run - {self.project}",
                            email_recipients,
                            {
                                "run_id": nemo_eval_id,
                                "run_status": "running",
                                "metrics": run_results,
                                "project_name": self.project
                            }
                        )
                        print("email_service_response", email_service_response)
                        results = {"notification": email_service_response, **run_results}

        except Exception as ex:
            raise f"Failed on finish Nemo eval run, {ex}"
        return results

    @log_errors("Prepare Nemo Eval")
    def prepare(
            self,
            request: RunMakerRequest,
            config: NVBotEvaluationConfig,
            project_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[DataFrame]:
        print('prepare nemo run')

        df = None
        assert request.Project is not None or request.ProjectId is not None, "Expect either project or projectId in request"
        if is_valid_instance(config.EvaluationSchema) and is_valid_instance(config.EvaluationSchema.NemoEvaluator):
            nemo_evaluator = config.EvaluationSchema.NemoEvaluator
            nemo_evaluators_entries = nemo_evaluator.Evaluators
            for entry in nemo_evaluators_entries:
                ds_content = None
                eval_type = None

                payload = entry["evaluator_payload"]
                model = entry.get("model")
                # prepare datastore asset
                eval_type = payload.get('eval_type')
                assert eval_type, f"Expect non-null eval_type {eval_type}"
                eval_type_definition = f"{eval_type}:{payload.get('eval_subtype')}" \
                    if payload.get('eval_subtype') is not None else f"{eval_type}"
                print("eval_types for nemo evaluator", eval_type_definition)
                logger.info(f"prepare for nemo evaluator eval_type: {eval_type_definition}")
                try:
                    dataset_config = nemo_evaluator.DatasetConfig
                    print('Datastore hosted at --> ', self.DS_URL)
                    # print('repo_name: ', self.name)
                    ds = NeMoDataStore(self.DS_URL, {"name": f"{self.name}-{eval_type}"})
                    # print(f"Get ds_content: {ds_content}")
                    print(f"Nemo dataset_config: {dataset_config}")
                    if dataset_config.Engine and dataset_config.Engine.lower() == StorageType.DATASTORE.value:
                        ds_content = {'DatasetId': dataset_config.DatasetId,
                                      'DatasetFolder': dataset_config.DatasetFolder,
                                      'Name': dataset_config.Name,
                                      'Files': dataset_config.Files
                                      }
                        ds.datastore_content = ds_content
                        print(f"Data config derived from config: {ds_content}")
                    if ds_content is None:
                        self.temp_folderpath = generate_nemo_eval_local_folder_path(request, eval_type)
                        # print("temp_folderpath: ", self.temp_folderpath)
                        if dataset_config.Engine.lower() == StorageType.LOCAL.value and dataset_config.RunFile:  # local
                            move_file(dataset_config.RunFile, self.temp_folderpath, CUSTOM_FILE_NAME)
                            local_file, local_file_ext = get_file_extension(dataset_config.RunFile)
                            # move_file(os.path.join(local_file, "log"), self.temp_folderpath, CUSTOM_FILE_NAME)

                        ds = self.trigger_datastore_creation_and_upload(eval_type, request, entry, dataset_config)

                        # NEED FIX: EVALUATOR SERVICE FIX
                        # this following block is for run_inference being False, which is only for custom eval
                        # if dataset_config.Engine and dataset_config.Engine.lower() == StorageType.LOCAL.value and \
                        #         eval_type.lower() == "automatic" and \
                        #         dataset_config.RunFile is not None \
                        #         and payload.get("inference_configs") is not None:
                            # print ("Check if need to upload bot answers")
                            # inference_configs = payload.get("inference_configs", [])

                            # if len(inference_configs) > 0:
                            #     run_inference = inference_configs[0].get("run_inference", True)
                            #     if run_inference is False:
                            #         logging.info(f"Upload bot answers to dataset {ds.datastore_content.get('id')}")
                            #         ds.upload_file(dataset_config.RunFile, "answers.xlsx")
                            #         # print ("complete", upload_answer_response)
                        if ds:
                            ds_content = ds.datastore_content

                    if ds is not None and ds_content is not None and self.validate_dataset_content(eval_type,
                                                                                                   ds_content):
                        print(f"dataset content: {ds_content}")
                        # only append evaluators when dataset is uploaded
                        eval_prepared = self.trigger_evaluator_creation(eval_type, ds_content, payload, model, request)

                        if eval_prepared:
                            self.evaluators.append(eval_prepared)
                            self.datastorers.append(ds)
                except Exception as ex:
                    logger.error(f"{ex} when creating dataset for {eval_type}")
                    if ds_content:
                        logger.error(f"{ex} when creating dataset content {ds_content}")

    @log_errors("Initiate Nemo Eval Result")
    async def initiate(self, request: RunMakerRequest, evaluation_history: dict):
        response = {}
        try:
            history_id = evaluation_history.get("id")
            param = UserEvaluationRunData(
                Id=history_id,
                Project=request.Project,
                NtAccount=EVAL_ACCOUNT_ID,
                Username=NT_ACCOUNT_NAME,
                Status=EvaluationRunStatus.COMPLETED.value,
                OutputUrl=evaluation_history.get("output_url", ""),
                RunType=evaluation_history.get("run_type", ""),
                EvalType=evaluation_history.get("eval_type", ""),
                Tag1=evaluation_history.get("tag1", ""),
                Tag2=evaluation_history.get("tag2", ""),
                Trace=evaluation_history.get("trace", ""),
            )
            update_response = DatabaseHandler({'env': 'dev'}).update_evaluation_history_and_details(param)
            # print("update database response: ", update_response)
            assert update_response, f"Failed to update database for history and details info: {update_response}"

            update_response = DatabaseHandler({'env': 'dev'}).delete_evaluation_history(history_id)
            # print("update database response: ", update_response)
            assert update_response, f"Failed to update database for history and details info: {update_response}"

            status_message = f"Updated evaluation history id: {evaluation_history.get('id')}"
            print(status_message)
        except Exception as ex:
            error_message = f"Exception in update evaluation history to database, {ex}"
            print(error_message)
        return response

    @log_errors("Trigger Nemo Evaluator Creation Service")
    def trigger_evaluator_creation(self, eval_type: str, dataset_config: dict,
                                   evaluator_config: dict, model_config: dict, request: RunMakerRequest):
        print(f"\nEval type: \n - {eval_type}\nDataset params:\n - {dataset_config}")
        if eval_type.lower() == "automatic":
            # NEED FIX: EVALUATOR SERVICE FIX
            return None
            eval_payload = prepare_custom_nemoeval_payload(self.EVAL_URL, dataset_config, evaluator_config,
                                                           model_config, request)
        else:
            eval_payload = prepare_llm_as_a_judge_nemoeval_payload(self.EVAL_URL, dataset_config, evaluator_config,
                                                                   model_config, request)
        logger.info(f"Eval type: {eval_type} with payload `{eval_payload}` created")
        print(f"\nEvaluation params:\n{eval_payload}")

        return NeMoEvaluator(self.EVAL_URL, eval_payload)

    @log_errors("Trigger Nemo Datastore Creation Service")
    def trigger_datastore_creation_and_upload(self, eval_type: str, request: RunMakerRequest,
                                              evaluator_config: dict, dataset_config: DatasetConfig):

        print("trigger_datastore_creation with load data from dataset_config", dataset_config)
        df = load_data_from_datasetconfig(self.project, dataset_config)
        data_limit = dataset_config.DataLimit
        if df is not None and data_limit is not None and df.shape[0] > data_limit:
            df = df[0:data_limit]
        assert df is not None, f"failed to load data from {dataset_config.Engine.lower()}"
        # prepare folder in local, naming format: script/nemo_eval/{request.Project}/{eval_type}-{str_datetime}
        if eval_type.lower() == "automatic":
            # prepare_custom_nemoeval_dataset(self.temp_folderpath, df, evaluator_config)
            return None
        elif eval_type.lower() == "llm_as_a_judge":
            prepare_llm_as_a_judge_nemoeval_dataset(self.temp_folderpath, df, evaluator_config)

        print(f"Prepare {eval_type} data ready: {self.temp_folderpath}")
        logger.info(f"Prepare {eval_type} data ready.")

        repo_name_unique = generate_unique_datastore_path(self.name, eval_type)
        ds = NeMoDataStore(self.DS_URL, {
            "name": repo_name_unique,
            "upload_dir": self.temp_folderpath
        })
        ds.create_via_hfapi()
        return ds

    @log_errors("Post-processing Nemo Eval Result")
    async def post_processing(self, evaluation_history: dict, nemo_eval_id: str):
        # custom eval download
        # e.g: <specified local dir>/automatic/custom_eval/results
        error = {}
        metrics = {}
        detail_evaluation_metrics = None
        print(f"Run against eval url -->  {self.EVAL_URL}")
        logger.info(f"Run against eval url -->  {self.EVAL_URL}")
        nemo_evaluator = NeMoEvaluator(self.EVAL_URL)
        evaluation_result = nemo_evaluator.get_results(nemo_eval_id)

        if evaluation_result.get("status") is None:
            # corner case: {'detail': 'Evaluation not found'}
            evaluation_result["status"] = "error"
            error_detail = evaluation_result.get("detail")
            evaluation_result["details"] = error_detail if error_detail else json.dumps(evaluation_result)

        evaluation_metrics = {}
        evaluation_status = evaluation_result["status"].lower()

        if evaluation_status not in ["succeeded", "failed", "error"]:
            return {"details": f"still running evaluation {nemo_eval_id}"}

        if evaluation_result.get("evaluations") is None:
            # return error message if failed to get evaluations
            evaluation_result["status"] = "error"
            evaluation_result[
                "details"] = f"Failed to get `evaluations` from evaluation_result, `{json.dumps(evaluation_result)}`"

        # update status to avoid race
        try:
            history_id = evaluation_history.get("id")
            param = UserEvaluationRunData(
                Id=history_id,
                Project=evaluation_history.get("project", None),
                NtAccount=EVAL_ACCOUNT_ID,
                Username=NT_ACCOUNT_NAME,
                Status='POST_PROCESSING',
                OutputUrl=evaluation_history.get("output_url", ""),
                RunType=evaluation_history.get("run_type", ""),
                EvalType=evaluation_history.get("eval_type", ""),
                Tag1=evaluation_history.get("tag1", ""),
                Tag2=evaluation_history.get("tag2", ""),
                Trace=evaluation_history.get("trace", ""),
            )
            update_response = DatabaseHandler({'env': 'dev'}).update_evaluation_history_and_details(param)
            # print("update database response: ", update_response)
            assert update_response, f"Failed to update database for history and details info: {update_response}"
            status_message = f"Updated evaluation history id: {evaluation_history.get('id')}"
        except Exception as ex:
            error_message = f"Exception in update evaluation history to database, {ex}"
            print(error_message)

        result_dataset_config = None
        eval_type = json.loads(evaluation_history.get("output_url")).get("eval_type", "") or ""

        # Note: needs to create with eval_id as datastore_content['id']
        ds = NeMoDataStore(self.DS_URL, {"id": nemo_eval_id})
        print("Evaluation_status: ", evaluation_status)

        def _construct_created_timestamp_str(created_time_datetime: str):
            utc_time = datetime.strptime(created_time_datetime, "%Y-%m-%dT%H:%M:%S")

            # Assign the UTC timezone to the datetime object
            utc_time = utc_time.replace(tzinfo=pytz.utc)

            # Convert the datetime object to US/Pacific time
            pacific_tz = pytz.timezone('US/Pacific')
            created_timestamp = utc_time.astimezone(pacific_tz)
            created_timestamp_str = created_timestamp.strftime('%Y-%m-%d_%H:%M:%S_%Z')
            return created_timestamp_str

        created_timestamp_str = _construct_created_timestamp_str(
            created_time_datetime=evaluation_result.get('created_at', "")
        )

        eval_metadata = evaluation_history.get("evaluation_metadatajson", {})
        eval_config = NVBotEvaluationConfig.model_validate(eval_metadata)
        nemo_eval_payload = get_evaluation_config_by_nemo_evaluator_type(eval_config.EvaluationSchema,
                                                                         LLM_AS_A_JUDGE_EVALUATOR_TYPE)
        model = nemo_eval_payload.get("model", {}) if nemo_eval_payload else {}
        env = json.loads(evaluation_history["metadata_value"]).get("Env") or self.env
        history_id = evaluation_history["id"]

        if evaluation_status.lower() == "succeeded":
            def _construct_result_filename(evaluation_history, datetime):
                # e.g: bot_config_name-env-datetime-metric_number
                bot_config_name = evaluation_history["project"]
                env = json.loads(evaluation_history["metadata_value"]).get("Env") or self.env

                if evaluation_history.get("tag1"):
                    tag_replacing_empty_space = re.sub(r" ", "-", evaluation_history.get("tag1", ""))
                    tag = re.sub(r"[^a-zA-Z0-9_ -]", "", tag_replacing_empty_space)
                    return f"{tag}-{evaluation_history.get('id')}-{bot_config_name}-{env}-{datetime}"
                return f"{evaluation_history.get('id')}-{bot_config_name}-{env}-{datetime}"

            result_filename = _construct_result_filename(evaluation_history, created_timestamp_str)

            eval_type, evaluation_status, evaluation_metrics, result_dataset_config, detail_evaluation_metrics = \
                await self.yield_eval_results(nemo_eval_id, evaluation_result, ds, evaluation_history, result_filename)

        # evaluation_run_status = EvaluationRunStatus.FAILED if evaluation_status == "failed" else EvaluationRunStatus.COMPLETED
        evaluation_run_status = EVAL_STATUS_MAPPING.get(evaluation_status.lower(), "")

        response = {"eval_id": nemo_eval_id, "eval_type": eval_type, "status": evaluation_status}
        print(f"🎉Eval Result response: {response}")
        logger.info(f"Result response {response}")
        print("evaluation_history:\n", evaluation_history)
        print("env:\n", self.env)
        evaluation_history_metadata = json.loads(evaluation_history.get("metadata_value"))

        bot_config_request = FlowConfigRequest.model_validate(evaluation_history_metadata)
        bot_name = "unknown"
        if self.env and bot_config_request.System and bot_config_request.Model:
            bot_config = await ConfigLoader(self.env).get_bot_config(bot_config_request)
            if bot_config:
                bot_name = bot_config.get("botName")
        elif bot_config_request.System:
            bot_name = bot_config_request.System
        else:
            bot_name = self.project

        dataset_metadatajson = evaluation_history.get("dataset_metadatajson")
        dataset_id = dataset_metadatajson.get("DatasetId", None)

        result_detail_url = ""
        if result_dataset_config is not None and evaluation_status in ["succeeded", "error"]:
            result_message = f"🏅Result dataset_config: {result_dataset_config}"
            print(result_message)
            logger.info(result_message)

            result_dataset_id = result_dataset_config.DatasetId
            result_file = (result_dataset_config.Files[0] if len(result_dataset_config.Files) > 0 else "").replace(" ",
                                                                                                                   "%2B")
            file_path_wrapped_by_braces = "{filepath}"
            if eval_type == 'automatic':
                result_detail_url = f"{self.settings.NVBOT_EVALUATION_URL}/datasets/download/{result_dataset_id}/file/contents/{file_path_wrapped_by_braces}?file_path=answers.xlsx"
                response["output_file"] = result_detail_url
            if eval_type == 'llm_as_a_judge':
                result_detail_url = f"{self.settings.NVBOT_EVALUATION_URL}/datasets/download/{result_dataset_id}/file/contents/{file_path_wrapped_by_braces}?file_path={result_file}"
                response["output_file"] = result_detail_url

        print(f"result dataset download url: {result_detail_url}")

        # Prepare notification data
        notification_data = {
            "bot_name": bot_name,
            "run_id": nemo_eval_id,
            "run_status": evaluation_status,
            "run_info": nemo_eval_id,
            "metrics": evaluation_metrics, # flatten_dict_by_1_degree(evaluation_metrics),
            "project_name": f"{self.project} on {self.env} environment created at {created_timestamp_str}",
            "result_url": result_detail_url,
            "summary_info": evaluation_result.get("details", None) if evaluation_status == "error" else None,
            "env": self.env,
            "creation_time": created_timestamp_str,
            "details": detail_evaluation_metrics,
            "tags":
                list(filter(lambda x: x is not None, [evaluation_history.get("tag1"), evaluation_history.get("tag2"),
                                                      evaluation_history.get("tag3")])),
            "run_url": f"{get_settings().NVBOT_EVALUATION_UI_URL}/dashboard/run?id={history_id}",
            "report": """      """
        }

        has_multiple_result = len(
            eval_config.EvaluationSchema.NemoEvaluator.Evaluators) > 1 and eval_type.lower() == CUSTOM_EVALUATOR_TYPE
        if not has_multiple_result:
            email_result = self.send_notification(notification_data)
            if email_result:
                response.update(email_result)

        try:
            # update history status
            assert history_id is not None, f"Expect id filed in evaluation history to be not None, got {history_id}"
            param = UserEvaluationRunData(
                Id=history_id,
                Project=evaluation_history.get("project", None),
                NtAccount=NT_ACCOUNT_ID,
                Username=NT_ACCOUNT_NAME,
                Status=evaluation_run_status.value,
                NemoEvalId=nemo_eval_id,
                OutputUrl=evaluation_history.get("output_url", ""),
                RunType=evaluation_history.get("run_type", ""),
                EvalType=evaluation_history.get("eval_type", ""),
                Tag1=evaluation_history.get("tag1", ""),
                Tag2=evaluation_history.get("tag2", ""),
                Trace=evaluation_history.get("trace", ""),
                Details=[UserEvaluationDetailData(
                    NtAccount=EVAL_ACCOUNT_ID,
                    Username=NT_ACCOUNT_NAME,
                    Metadata=json.dumps(response),
                    EvaluatorType=str(eval_type),
                    Criteria="",
                    Metrics=json.dumps(evaluation_metrics) if evaluation_metrics else "{}",
                    Annotation=json.dumps(evaluation_result.get("details")) if evaluation_result.get(
                        "details") else None,
                    ResultMetadata={"url": result_detail_url}
                )]
            )

            update_response = DatabaseHandler({'env': 'dev'}).update_evaluation_history_and_details(param)
            # print("update database response: ", update_response)
            assert update_response, f"Failed to update database for history and details info: {update_response}"
            assert update_response.get(
                "status").lower() == evaluation_run_status.value.lower(), "status is not updated"

            status_message = f"Updated evaluation history id: {evaluation_history.get('id')}"
            print(status_message)
        except Exception as ex:
            error_message = f"Exception in update evaluation history to database, {ex}"
            print(error_message)
            response = {"details": error_message, **response}
            return response
        print(f"Nemo Evaluation Result: {nemo_eval_id} - {evaluation_status}")

        if result_dataset_config:
            response = {
                **{"result_dataset_config": result_dataset_config.dict(exclude_none=True)},
                **response,
            }
        print(f"response: {response}")

        tags = [
            f"history-id:{history_id}",
            f"eval-id:{nemo_eval_id}",
            f"project-name:{self.project}",
            f"bot-name:{bot_name or ''}",
            f"dataset-id:{dataset_id}",
            f"environment:{self.env}",
            f"model:{model.get('llm_name') or ''}",
            f"creation_time:{created_timestamp_str}"
        ]
        print(f"Notification data: {notification_data}")
        print(f"Notification tags: {','.join(tags)}")
        logger.info(f"Notification data: {notification_data}")
        logger.info(f"Notification tags: {','.join(tags)}")

        eval_config = NVBotEvaluationConfig.model_validate(eval_metadata)

        if evaluation_history.get("run_type", "").lower() == "cron":
            self.send_metrics_to_datadog(notification_data=notification_data, tags=tags)

        return response

    @log_errors("Send Metrics to Datadog")
    def send_metrics_to_datadog(self,
                                notification_data: dict,
                                tags: List[str]
                                ):

        # parse data
        flatten_metrics = convert_to_snake_case_separated_by_dot_notation(notification_data.get("metrics"))
        DatadogHandler().send_eval_metrics(notification_data, flatten_metrics, tags)

    @log_errors("Send Notification")
    def send_notification(self, notification_data: dict):
        allow_notification = self.config.EvaluationSchema.Notification
        email_recipients = allow_notification.EmailRecipients or []
        email_recipients += ["yangj@nvidia.com"] if "yangj@nvidia.com" not in [] else []

        if allow_notification:
            try:
                logger.info(f"email recipients: {email_recipients}")
                if True:
                    status_icon = "✅" if notification_data["run_status"].lower() == "succeeded" \
                        else "⚠️" if notification_data["run_status"].lower() == "failed" else "🤔"
                    email_service_response = EmailServices().send_email(
                        f"{status_icon} reports - {notification_data['project_name']}",
                        email_recipients,
                        notification_data
                    )
                    logger.info(f"email_service_response: {email_service_response}")
                    return {"notification": email_service_response}
            except Exception as ex:
                logger.error(f"error when sending email notification: {ex}")

    @log_errors("Validate dataset content")
    def validate_dataset_content(self, eval_type: str, ds_config: dict):
        # TODO: add content check
        files = ds_config.get("Files", [])
        if files:
            if eval_type.lower() == "automatic":
                input_file = [path for path in files if
                              path.endswith("input.jsonl") or path.endswith(CUSTOM_EVAL_INPUT_FILENAME)]

                return len(input_file) == 1
            elif eval_type.lower() == "llm_as_a_judge":
                judge_prompt_file = [path for path in files if path.endswith("judge_prompts.jsonl")]
                question_file = [path for path in files if path.endswith("question.jsonl")]

                return len(judge_prompt_file) == 1 and len(question_file) == 1
            return False

    @log_errors("Yield Evaluation Results")
    async def yield_eval_results(self, eval_id: str, evaluation_result: dict, datastore: NeMoDataStore,
                                 evaluation_history: dict, result_filename: Optional[str]):
        print(f"Start fetching eval results for: {eval_id}")
        evaluations = evaluation_result.get("evaluations", [])
        eval_type = ""
        status = "error"
        metrics = {}
        result_dataset_config = None
        eval_results = None

        if evaluations and len(evaluations) > 0:
            # default to only process one/first
            eval_type = evaluations[0].get("eval_type")
            eval_subtype = evaluations[0].get("eval_subtype")
            logger.info(f"Process Evaluation Results for '{eval_type}' on '{evaluation_history.get('project')}")
            if eval_type.lower() == LLM_AS_A_JUDGE_EVALUATOR_TYPE:
                eval_results = await yield_llm_eval_results(eval_id, evaluation_result, evaluation_history, datastore,
                                                            self.config, result_filename)
            elif eval_type.lower() == CUSTOM_EVALUATOR_TYPE:
                eval_results = yield_custom_eval_results(evaluation_result,
                                                         evaluation_history, result_filename)
            if eval_results is not None:
                status, metrics, result_dataset_config, detail_metrics = eval_results
                return eval_type, status, metrics, result_dataset_config, detail_metrics
        else:
            error_message = "Unable to get eval_type in evaluation results from `evaluation` payload"
            print(error_message)
            logging.error(error_message)
        return eval_type, status, metrics, result_dataset_config, None


@log_errors("Yield Custom evaluation results")
def yield_custom_eval_results(evaluation_result: dict, evaluation_history: dict, result_filename: Optional[str]) -> \
        Tuple[
            str, dict, DatasetConfig | None, Any | None]:
    evaluation_status = evaluation_result["status"].lower()
    metrics = {}
    dataset_metadata = evaluation_history['dataset_metadatajson']
    dataset_id = dataset_metadata.get("DatasetId")

    if evaluation_result['evaluation_results'] and len(evaluation_result['evaluation_results']) > 0:
        raw_metrics = evaluation_result['evaluation_results'][0].get("aggregated_results") if \
            evaluation_result['evaluation_results'][0] else {}
        metrics = parse_custom_eval_aggregated_metrics(raw_metrics)
        logger.info(f"metrics: {metrics}")
        # print(metrics)
        return evaluation_status, metrics, DatasetConfig(
            Engine=StorageType.DATASTORE.value,
            DatasetId=dataset_id,
            Files=[CUSTOM_FILE_NAME]
        ), None
    logger.error(f"Failed to get custom evaluation_results : {dataset_id}")
    return evaluation_status, {}, None, None

    # if evaluation_result['evaluation_results'] and len(evaluation_result['evaluation_results']) > 0:
    #     complete_metrics = evaluation_result['evaluation_results'][0].get("metrics") if \
    #         evaluation_result['evaluation_results'][0] else {}
    #     return evaluation_status, parse_llm_as_a_judge_eval_aggregated_metrics(complete_metrics)


@log_errors("Yield LLM evaluation results")
async def yield_llm_eval_results(eval_id: str, evaluation_result: dict, evaluation_history: dict,
                                 datastore: NeMoDataStore,
                                 eval_config: NVBotEvaluationConfig, result_filename: Optional[str]) -> Tuple[
    str, dict, DatasetConfig, Any | None]:
    evaluation_status = evaluation_result["status"].lower()
    metrics = {}
    eval_type = LLM_AS_A_JUDGE_EVALUATOR_TYPE.lower()
    directory = LOCAL_EVAL_RESULTS_TMP_FOLDER
    create_folder_if_not_exists(directory)
    results_directory = datastore.download_results_to_local_directory(eval_id, directory)
    nemo_eval_payload = get_evaluation_config_by_nemo_evaluator_type(eval_config.EvaluationSchema,
                                                                     LLM_AS_A_JUDGE_EVALUATOR_TYPE)

    assert results_directory is not None, f"Failed to download evaluation result by {eval_id}"

    # download eval result and files by nemo_eval_id
    def _find_nested_file_by_name_and_filetype(base_directory: str, folder_: str = None, file_: str = None,
                                               filetype_: str = None) -> str:
        """
        Recursively search for the model_judgement directory and find the jsonl file.
        """
        for root, dirs, files in os.walk(base_directory):
            # Exclude hidden folders
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            if folder_ is not None and folder_ in dirs:
                current_dir = os.path.join(root, folder_)
                # print("dir ", current_dir, os.listdir(current_dir))
                for file_name in os.listdir(current_dir):
                    filetype_suffix_matches = filetype_ is not None and file_name.endswith(filetype_)
                    if file_ is None:
                        return os.path.join(current_dir, file_name)
                    elif file_ is not None and filetype_suffix_matches and file_.lower() in file_name.lower():
                        return os.path.join(current_dir, file_name)
            else:
                # folder is None, should check matched file_
                for folder in dirs:
                    current_dir = os.path.join(root, folder)
                    for file_name in os.listdir(current_dir):
                        filetype_suffix_matches = filetype_ is not None and file_name.endswith(filetype_)
                        if file_ is not None and filetype_suffix_matches and file_.lower() in file_name.lower():
                            return os.path.join(current_dir, file_name)

    # Finding the JSONL file in the directory
    answer_jsonl_file = _find_nested_file_by_name_and_filetype(results_directory, 'model_answer', None, ".jsonl")
    if not answer_jsonl_file:
        raise FileNotFoundError("No JSONL file found in the model_answer folder")

    scores_jsonl_file = _find_nested_file_by_name_and_filetype(results_directory, 'model_judgment', None, ".jsonl")
    if not scores_jsonl_file:
        raise FileNotFoundError("No JSONL file found in the model_judgement folder")

    question_jsonl_file = _find_nested_file_by_name_and_filetype(results_directory, None, "question", ".jsonl")
    if not question_jsonl_file:
        raise FileNotFoundError("No JSONL file found matching question.jsonl file")

    reference_jsonl_file = _find_nested_file_by_name_and_filetype(results_directory, "reference_answer", "reference",
                                                                  ".jsonl")
    if not reference_jsonl_file:
        raise FileNotFoundError("No JSONL file found matching reference_answer.jsonl file")

    # fetch jsonl file under model_judgement folder, there should only has 1 jsonl file, check it
    eval_scores = nemo_eval_payload.get("judge_config", {}).get("scorers")
    column_map = nemo_eval_payload.get("column_map", {})
    category_column = column_map.get("category")
    run_config = nemo_eval_payload.get("run_config")

    if eval_scores is None:
        scores = {
            "Correctness Answer": [],
            "Helpfulness": [],
            "Empathy": [],
            "Conciseness": []
        }
    else:
        scores = {score: [] for score in eval_scores}

    parse_keys = nemo_eval_payload.get("judge_config", {}).get("parse_keys")
    if parse_keys is None:
        parse_keys = [
            "Explanation"
        ]

    report_metrics = {}
    answer_xlsx_file = _find_nested_file_by_name_and_filetype(results_directory, None,
                                                              os.path.splitext(ANSWER_FILENAME)[0], ".xlsx")
    dataset_df = file_content_convertor(output_format='pd', file=answer_xlsx_file, parse_dates=False)
    answer_xlsx_sheet_names = list(pd.ExcelFile(answer_xlsx_file).sheet_names)

    latency_metrics = None
    verbosity_metrics = None
    status_metrics = None
    metrics, quality_metrics = {}, {}

    if "Latency" in answer_xlsx_sheet_names:
        latency_df = file_content_convertor(output_format='pd', file=answer_xlsx_file, parse_dates=False,
                                            sheet_name="Latency").to_dict(orient='index')
        latency_metrics = {}
        for a_key, a_row in latency_df.items():
            copy_row = a_row.copy()
            del copy_row["Unnamed: 0"]
            latency_metrics[a_row.get("Unnamed: 0", "_key")] = copy_row

    if "Verbosity" in answer_xlsx_sheet_names:
        verbosity_df = file_content_convertor(output_format='pd', file=answer_xlsx_file, parse_dates=False,
                                              sheet_name="Verbosity").to_dict(orient='index')
        verbosity_metrics = {}
        for a_key, a_row in verbosity_df.items():
            copy_row = a_row.copy()
            del copy_row["Unnamed: 0"]
            verbosity_metrics[a_row.get("Unnamed: 0", "_key")] = copy_row

    if "Status" in answer_xlsx_sheet_names:
        status_df = file_content_convertor(output_format='pd', file=answer_xlsx_file, parse_dates=False,
                                           sheet_name="Status").to_dict(orient='index')

        status_metrics = {}
        for a_key, a_row in status_df.items():
            copy_row = a_row.copy()
            del copy_row["Unnamed: 0"]
            status_metrics[a_row.get("Unnamed: 0", "_key")] = copy_row
        status_error_rate = round(
            1 - list(status_metrics.get('200').values())[0] / list(status_metrics.get('Total').values())[0], 2)
        status_metrics['Error rate'] = {'Status Distribution': status_error_rate}

    logger.info(f"status_metrics: {status_metrics}")
    print(f"status_metrics: {status_metrics}")
    metrics["Status Distribution"] = status_metrics

    rows = {}
    for index, row in dataset_df.iterrows():
        row_dict = row.to_dict()
        rows[index] = row_dict

    answer_response = []

    with open(answer_jsonl_file, 'r') as file, open(question_jsonl_file, 'r') as q_file, open(reference_jsonl_file,
                                                                                              'r') as r_file:
        for index, (line1, line2, line3) in enumerate(zip(file, q_file, r_file)):
            raw_data = json.loads(line1.strip())
            raw_question_data = json.loads(line2.strip())
            raw_reference_data = json.loads(line3.strip())
            extracted_data = {
                # "Question_id": raw_data["question_id"],
                **rows[index],
                # "Question": raw_question_data["turns"][0],
                "Response": raw_data["choices"][0]["turns"][0],
                # "Reference": raw_reference_data["choices"][0]["turns"][0],
            }
            # print (f"\n{extracted_data}")
            answer_response.append(extracted_data)

    # then for each line in jsonl, call helper function to parse score value
    # after parse, accumulate scores by the value it is given, accumulate into a list of scores
    contains_unable_parsing_scores = False
    parser = None
    if run_config is not None and run_config.get("Outputs", None) is not None:
        parser = await dict_parser(parser_config=run_config.get("Outputs"))

    with open(scores_jsonl_file, 'r') as file:
        for index, line in enumerate(file):
            line_json = json.loads(line.strip())
            extracted_scores = {key: None for key in scores.keys()}

            extracted_text_from_judgement = extract_string_from_llm_judgment(line_json.get("judgment"), parse_keys)

            extracted_scores.update(extracted_text_from_judgement)

            extracted_scores.update({"Judgement Raw Response": line_json.get("judgment")})
            extracted_numeric_values = extract_numeric_value_from_llm_judgment(line_json.get("judgment"))

            if len(list(extracted_numeric_values.keys())) == 0:
                contains_unable_parsing_scores = True

            if parser is not None:
                try:
                    parser_result = (await parser({
                        "values": extracted_numeric_values,
                        "row": answer_response[index]
                    }))
                    row = parser_result.get("row", answer_response[index])
                    extracted_numeric_values = parser_result.get("values", extracted_numeric_values)
                    answer_response[index] = row
                except Exception as parser_exception:
                    logger.error(f"Parser exception: {parser_exception}")
                    print(f"Parser exception: {parser_exception}")

            # override scorers to 0 if Response is null
            row_response = answer_response[index].get("Response")
            row_status_code = int(answer_response[index].get("Status Code", -1)) if (str(answer_response[index].get("Status Code"))).isdigit() else -1
            if not row_response and row_status_code == 200:
                extracted_numeric_values.update({key: 0 for key in scores})

            for extracted_key in extracted_numeric_values:
                if extracted_key in scores:
                    # override numeric scores
                    scores[extracted_key].append(extracted_numeric_values.get(extracted_key, None))

                extracted_scores[extracted_key] = extracted_numeric_values.get(extracted_key, "")
            answer_response[index].update(extracted_scores)

    result_filename = result_filename or LLM_AS_A_JUDGE_FILE_NAME

    for key, values in scores.items():
        filtered_values = [v for v in values if v is not None]

        if filtered_values:
            quality_metrics[key] = {
                "Mean": round(statistics.mean(filtered_values), 2),
                "Std": round(statistics.stdev(filtered_values), 2) if len(filtered_values) > 1 else 0
            }
        if key in quality_metrics:
        # metric_data = convert_to_snake_case_seperated_by_dot_notation(metrics)
            result_filename += f"-{key.lower().replace(' ', '')}{quality_metrics[key].get('Mean', '')}"

    logger.info(f"quality_metrics: {quality_metrics}")
    print(f"quality_metrics: {quality_metrics}")
    metrics["Quality Metrics"] = quality_metrics

    category_metrics = calculate_category_metrics(
        category_column, answer_response, list(scores.keys()))
    if category_metrics:
        print("category_metrics", category_metrics)
        report_metrics["Category Metrics"] = category_metrics

    # add consistency checks
    comparison_run_results = None
    trends_metrics = None
    if is_valid_instance(eval_config.ComparisonSchema):
        cc = ComparisonContainer(
            project=evaluation_history.get("project"),
            config=eval_config,
        )
        comparison_config = cc.build_datasetconfigs(evaluation_history)

        evaluation_history_metadata = json.loads(evaluation_history.get("metadata_value"))

        if comparison_config:
            comparison_config.ComparisonSchema.DatasetConfigs.append(
                DatasetConfig(
                    Engine=StorageType.DATAFRAME.value,
                    Data=answer_response
                )
            )
            run_request = RunMakerRequest(
                Project=evaluation_history.get("project"),
                ProjectId=evaluation_history.get("project_id", None),
                RunType=evaluation_history.get("run_type"),
                UserId=evaluation_history_metadata.get("UserId", ""),
                UserName=evaluation_history_metadata.get("UserName", ""),
                Parameters={"tag1": evaluation_history.get("tag1"), "tag2": evaluation_history.get("tag2")},
                EvaluationConfig=comparison_config,
                System=FlowConfigRequest.model_validate(evaluation_history_metadata).System,
                Model=FlowConfigRequest.model_validate(evaluation_history_metadata).Model,
                Env=FlowConfigRequest.model_validate(evaluation_history_metadata).Env
            )
            prepared_df = await cc.prepare(request=run_request, config=comparison_config)
            comparison_run_results = await cc.arun(request=run_request, config=comparison_config, df=prepared_df)

            comparison_results = comparison_run_results.get("results_df")
            trends_metrics = comparison_run_results.get("trends_metrics").to_dict()
            # report_metrics["Trend metrics"] = trends_metrics

    # upload answer to eval result folder for direct access
    output_file = os.path.join(directory, eval_id, eval_type.lower(),
                               result_filename)
    if comparison_run_results and trends_metrics:

        comparison_results = comparison_run_results.get("results_df")

        file_path = generate_excel_output(
            output_file,
            ['Results', {'Comparision': {
                "index": cc.get_outlier_row_index(comparison_results),
                "color": "#ffd966"
            }}],
            [pd.DataFrame(answer_response), comparison_run_results.get("results_df")],
            ["Metrics", "Category Metrics", "Latency", "Verbosity", "Trend"],
            eval_metrics=quality_metrics,
            category_metrics=category_metrics,
            latency_metrics=latency_metrics,
            verbosity_metrics=verbosity_metrics,
            trends_metrics=trends_metrics
        )
    else:
        file_path = generate_excel_output(
            output_file,
            ['Results'],
            [pd.DataFrame(answer_response)],
            ["Metrics", "Category Metrics", "Latency", "Verbosity", "Status"],
            eval_metrics=quality_metrics,
            category_metrics=category_metrics,
            latency_metrics=latency_metrics,
            verbosity_metrics=verbosity_metrics,
            status_metrics=status_metrics
        )

    print(f"📁File path: {file_path}")

    dataset_name = f"{eval_id}-results-{generate_short_uuid(5)}"

    dir_name, base_name = os.path.dirname(file_path), os.path.basename(file_path)
    upload_result_id, path = create_dataset_and_upload_folder(datastore.url, dataset_name, dir_name, path_in_repo=".")
    print("upload answers response: ", upload_result_id)
    logger.info(f"Processed LLM eval results uploaded to datastore: {upload_result_id}")

    if latency_metrics:
        metrics.update({"Latency Metrics": latency_metrics})

    return "error" if contains_unable_parsing_scores else evaluation_status, metrics, DatasetConfig(
        Engine=StorageType.DATASTORE.value,
        DatasetId=upload_result_id,
        Files=[base_name],
        RunFile=file_path
    ), report_metrics


@log_errors("Save evaluation results")
def _save_evaluation_history_to_database(request: RunMakerRequest, datastore_content: dict,
                                        evaluation_result: Any) -> None:
    response = None
    try:
        evaluation_history_id = DatabaseHandler({'env': 'dev'}).add_to_evaluation_history(
            run_request=request,
            dataset_metadata=datastore_content,
            evaluation_metadata=request.EvaluationConfig.model_dump(
                exclude_unset=True) if request.EvaluationConfig else {},
            flowconfig_metadata=request.PlatformConfig.model_dump(exclude_unset=True) if request.PlatformConfig else {},
            output_url=json.dumps(evaluation_result)
        )
        print(f"Saved evaluation history in database: {evaluation_history_id}")
        logger.info(f"Saved evaluation history in database: {evaluation_history_id}")

        # verify
        response = DatabaseHandler({'env': 'dev'}).get_evaluation_history(
            history_id=int(evaluation_history_id),
        )
        print("🍵 Save evaluation results response: ", response)
        # print("Save evaluation results evaluation_result: ", evaluation_result)
        return response
    except Exception as ex:
        print(f"Exception in save evaluation history to database, {ex}")
        raise Exception(ex)
    return response
