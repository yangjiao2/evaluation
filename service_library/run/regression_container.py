from __future__ import annotations

import asyncio
import concurrent.futures
import json
from ddtrace import tracer
from starlette_context import context

import logging
import pprint
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union, cast, Callable
from uuid import UUID, uuid4

import aiohttp
import numpy as np
import pandas as pd
import pytz
import requests
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers import EvaluatorCallbackHandler

from langsmith.evaluation import EvaluationResult
from pandas import DataFrame
from tqdm import tqdm
from configs.settings import (
    get_settings,
    get_cache_session,
    get_chatbot_url,
    get_graph_chatbot_url,
)
from controllers.auth.auth_token_loader import AuthTokenLoader
from data_models.api.evaluation_result import RegressionRunMetrics
from data_models.dataset_handler import DataContentOutputConfig
from data_models.api.run_maker import (
    RunMakerRequest,
    NVBotEvaluationConfig,
    StorageType,
)

from nvbot_models.request_models.evaluation_request import EvaluationRunStatus

# from nv_platform.nvbot_platform import NVBotPlatform
from nvbot_models.request_models.fulfillment_request import FulfillmentRequest
from service_library.constants import LOCAL_EVAL_RESULTS_TMP_FOLDER
from service_library.handler.database_handler import DatabaseHandler
from service_library.handler.s3_dataset_handler import S3DatasetHandler
from service_library.nemo_ms.nemo_service_helper import get_extract_run_maker_request
from service_library.notification.send_email import EmailServices
from service_library.run.parser_library.input_parser import (
    format_chat_request_post_params,
)
from service_library.run.parser_library.response_parser import (
    fetch_post_chatbot_request_response,
)
from service_library.run.parser_library.dict_parser import dict_parser
from service_library.run.run_container import RunContainer
from service_library.url_wrappers.url_wrapper import URLWrapper
from service_library.utils.api_helper import post_json
from service_library.utils.data_helper import (
    drop_df_empty_columns,
    safe_ast_literal_eval,
)
from service_library.utils.datasetconfig_helper import load_data_from_datasetconfig
from service_library.utils.header_composer import header_auth_composer
from service_library.utils.logging import log_errors
from service_library.utils.pydantic_helper import is_valid_instance
from service_library.utils.request_helpers import create_header, evict_jwk_token
from service_library.utils.response_checker.verbosity_validation import (
    get_verbosity_statistics_comparison,
)
from service_library.utils.run_helpers import (
    convert_to_snake_case,
    generate_excel_output,
    create_folder_if_not_exists,
)

# from utility import generate_regression_filename

logger = logging.getLogger(__name__)

DEFAULT_COLUMN_MAP = {
    "question": "Query",
    "reference": "Correct Answer",
    "answer": "Response",
    "category": "Source",
}


class RegressionRunContainer(RunContainer):
    """A container to help manage the state of an eval run."""

    input_mapper = None
    output_mapper = None

    def __init__(self, project: str, config: NVBotEvaluationConfig, env: str):
        super().__init__(project, config, env)
        self.chatbot_url = get_chatbot_url(env)  # default

        self.input_mapper = None
        self.output_mapper = None

        self.metrics = RegressionRunMetrics()
        self.metrics_name = "Latency"

        self.temp_filepath = None
        evict_jwk_token(env)

    @log_errors("Initiate")
    async def initiate(self, request: RunMakerRequest):
        return _save_evaluation_history_to_database(
            request, EvaluationRunStatus.IN_PROCESS
        )

    @log_errors("Answer generation")
    async def arun(
        self,
        request: RunMakerRequest,
        config: NVBotEvaluationConfig,
        df: DataFrame,
        **kwargs,
    ):
        print("Regression arun")
        headers = create_header(self.auth_token)

        results = []
        self.metrics.Total = df.shape[0]
        self.temp_filepath = self._generate_excel_filepath(request)
        config_type = (
            request.PlatformConfig.ConfigType.lower()
            if request.PlatformConfig and request.PlatformConfig.ConfigType
            else ""
        )

        if config_type.lower() == "api":
            headers, auth = header_auth_composer(request.PlatformConfig.Auth, {})

            chat_request_params = {
                "chatbot_url": request.PlatformConfig.URL,
                "chatbot_request": request.PlatformConfig.Payload or {},
                "chatbot_header": request.PlatformConfig.Header,
            }
            self.chatbot_url = request.PlatformConfig.URL
        else:
            chat_request_params = await format_chat_request_post_params(request, "")
            self.chatbot_url = chat_request_params.get("chatbot_url")

        column_map = (
            config.RegressionSchema.DataConfigs.get("column_map")
            if config.RegressionSchema.DataConfigs
            else DEFAULT_COLUMN_MAP
        )
        question_column = column_map.get("question", DEFAULT_COLUMN_MAP.get("question"))
        category_column = column_map.get("category", DEFAULT_COLUMN_MAP.get("category"))
        answer_column = column_map.get("answer", DEFAULT_COLUMN_MAP.get("answer"))

        chatbot_request = chat_request_params.get("chatbot_request")
        # optional
        reference_column = column_map.get(
            "reference", DEFAULT_COLUMN_MAP.get("reference")
        )

        status_code_distributions = {}
        latencies = []
        failures = []
        print("Platform run at --> ", self.chatbot_url)

        span = tracer.current_span()
        logger.info(f"regression_arun span: {span}")
        print(f"regression_arun span : {span}")
        trace_id = str(span.trace_id) if span else "None"
        # logger.info(f"regression_arun trace_id: {trace_id}")
        # print(f"regression_arun trace_id : {trace_id}")

        context["trace_id"] = str(trace_id)

        # async with aiohttp.ClientSession(timeout=timeout) as request_session:
        if self.chatbot_url and chat_request_params:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                time.sleep(10)

                question = row.get(question_column, "")
                processed_result = {question_column: question}
                retry_count, duration_avg = "", 0

                try:
                    self.metrics.Processed += 1
                    start = time.time()
                    # print(f"-- #{index + 1} \nstart: ", datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S'))
                    # TODO: build request parameters (prepare_chatbot_request)
                    row_dict = row.to_dict()

                    # NOTE: leave it JUST for UAT testing
                    # if row["feedback"].lower() != "bad":
                    #     parsed_response = {"response": ""}
                    # else:
                    parsed_response = None
                    chatbot_response, status_code = None, 500
                    # print("config_type: ", config_type)

                    if config_type.lower() == "api":
                        request_dict = await RegressionRunContainer.input_mapper(
                            row_dict
                        )
                        request_payload = {
                            r_key: safe_ast_literal_eval(r_value)
                            for r_key, r_value in request_dict.items()
                            if r_value
                        }
                        request_payload.update(chatbot_request)
                        # request_payload["Query"] = question
                        # request_payload["Category"] = "hybrid_search"
                        headers, auth = header_auth_composer(
                            request.PlatformConfig.Auth, row
                        )
                        with tracer.trace(
                            "chatbot.request", service="nvbot-evaluator"
                        ) as span:
                            trace_id = span.trace_id
                            span_id = span.span_id
                            # logger.info(f"chatbot.request trace_id: {trace_id}")
                            print(f"chatbot.request trace_id : {trace_id}")
                            (
                                api_response,
                                status,
                                status_code,
                                retry_count,
                                duration_avg,
                            ) = post_json(
                                self.chatbot_url, headers, request_payload, auth
                            )

                        chatbot_response = (
                            api_response.json()
                            if status_code == 200
                            else api_response.text
                        )

                    else:
                        chatbot_request.Query = question

                        logger.info(f"#{index + 1} Query: {chatbot_request.Query}")
                        # with tracer.trace("chatbot.request", service="nvbot-evaluator") as span:
                        #     trace_id = span.trace_id
                        #   span_id = span.span_id
                        # headers["x-datadog-trace-id"] = str(trace_id)
                        # headers["x-datadog-parent-id"] = str(span_id)
                        # logger.info(f"chatbot.request trace_id: {trace_id}")
                        # print(f"chatbot.request trace_id : {trace_id}")

                        parsed_response, retry_count, duration_avg = (
                            await fetch_post_chatbot_request_response(
                                chatbot_request,
                                self.chatbot_url,
                                request.Env,
                                row_dict.get("Context"),
                            )
                        )

                        logger.info(f"#{index + 1} Parsed response: ", parsed_response)
                        status_code = parsed_response.get("status")

                        if parsed_response and parsed_response.get("response"):
                            status_code = parsed_response.get("status")
                            chatbot_response = parsed_response.get("response")

                    logger.info(f"#{index + 1} Response status: {status_code}")
                    # print(f"#{index + 1} \n Response status: {status_code},  Query: {chatbot_request.Query}, Response: {chatbot_response}")
                    assert (
                        chatbot_response is not None
                    ), f"Failed to get chatbot response {question}"

                    status_code_distributions[f"{status_code}"] = (
                        status_code_distributions.get(f"{status_code}", 0) + 1
                    )

                    end = time.time()
                    print(
                        f"-- #{index + 1}  end: { datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S')}"
                    )

                    if status_code == 200 and chatbot_response is not None:
                        # for direct response
                        if isinstance(chatbot_response, str):
                            processed_result["Response"] = chatbot_response
                            # processed_result = result
                        # for structured response
                        elif isinstance(chatbot_response, list):
                            parser_result = await RegressionRunContainer.output_mapper(
                                chatbot_response
                            )
                            processed_result = parser_result

                        elif isinstance(chatbot_response, dict):
                            # status = parsed_response.get("status")
                            # if int(status) != 200:
                            processed_result = (
                                await RegressionRunContainer.output_mapper(
                                    {**chatbot_response, **processed_result}
                                )
                            )
                            # processed_result = {**result, **parser_result}

                        if not processed_result.get("Duration"):
                            processed_result["Duration"] = duration_avg

                            latencies.append(end - start)
                        else:
                            latencies.append(
                                round(float(processed_result.get("Duration")), 2)
                            )
                        # processed_result["IsStream"] = str(chatbot_request.Parameters.IsStream)
                        # processed_result["Request"] = json.dumps(chatbot_request.dict())
                        # pprint.pprint(chatbot_request.dict())
                        res = str(processed_result.get(answer_column, ""))

                        # TEMP FIX: temp lazy handling
                        if request.System and request.System.lower() in [
                            "nvinfo",
                            "nvhelp",
                            "scout",
                            "globalprotect",
                        ]:
                            processed_result["URL validation"] = (
                                URLWrapper().validate_urls(res)
                            )

                    else:
                        processed_result["Faulty Status Text"] = chatbot_response
                        processed_result["Faulty Duration"] = duration_avg
                        failures.append(processed_result)

                except asyncio.TimeoutError as te:
                    print(f"The request timed out")
                    processed_result["Error"] = str(te)

                except Exception as ex:
                    message = f"Exception in regression run on #{index} Query {row[question_column]}: {ex}"
                    print(message)
                    processed_result["Error"] = str(ex)

                # mark as "Faulty response" if response is empty, after checking "Response"
                if (
                    "Response" in processed_result
                    and processed_result.get("Response") == ""
                ):
                    status_code_distributions["Faulty Response"] = (
                        status_code_distributions.get("Faulty Response", 0) + 1
                    )

                processed_result["Status Code"] = status_code
                if retry_count > 0:
                    processed_result["# Retry"] = retry_count

                results.append(processed_result)

                if processed_result:
                    for key, value in processed_result.items():
                        if isinstance(value, (str, float, int, np.ScalarType)):
                            df.loc[index, key] = value
                        elif isinstance(value, (dict, list)):
                            df.loc[index, key] = str(value)

                else:
                    error_message = f"🚨Failed to get response from {self.chatbot_url}, status: {parsed_response}"
                    logger.error(error_message)
                    print(f"\n{error_message}\n")

                    # raise Exception(message)
            print("latencies: ", latencies)
            assert (
                len(latencies) > 0
            ), f"🚨Errors received on create chatbot response from {self.chatbot_url}, please validate request params."

            self.metrics.Avg = round(np.average(latencies), 2)
            self.metrics.P50 = round(np.percentile(latencies, 50), 2)
            self.metrics.P90 = round(np.percentile(latencies, 90), 2)
            self.metrics.P95 = round(np.percentile(latencies, 95), 2)
            self.metrics.P99 = round(np.percentile(latencies, 99), 2)
            self.metrics.Std = round(
                np.std(
                    latencies,
                ),
                2,
            )
            print("Regression Metric: ", self.metrics.dict(exclude_none=True))

            df = drop_df_empty_columns(df, ["Duration", "URL validation"])
            # latency metric
            latency_metrics = self.metrics.dict(exclude_none=True)

            status_code_distributions["Total"] = latency_metrics["Total"]

            del latency_metrics["Total"]
            del latency_metrics["Processed"]

            verbosity_metrics = None
            print(
                "Verbosity checks: ",
                answer_column in df.columns,
                reference_column,
                reference_column in df.columns,
            )
            if answer_column in df.columns and reference_column in df.columns:
                # verbosity metric

                verbosity_metrics = get_verbosity_statistics_comparison(
                    df[answer_column].tolist(), df[reference_column].tolist(), ""
                )

            generate_excel_output(
                self.temp_filepath,
                ["Results"],
                [df],
                ["Latency", "Verbosity", "Status"],
                latency_metrics={self.metrics_name: latency_metrics},
                verbosity_metrics=verbosity_metrics,
                status_code_distribution={
                    "Status Distribution": status_code_distributions
                },
            )
        return {"local_filepath": self.temp_filepath, "results": results}

    # def _collect_metrics(self, results):

    @log_errors("Finish run")
    def finish(
        self,
        request: RunMakerRequest,
        config: NVBotEvaluationConfig,
        run_results: dict = {},
    ):
        dataset_config = config.RegressionSchema.DatasetConfig
        allow_notification = config.RegressionSchema.Notification
        dataset_folder = dataset_config.Name
        result_folder = dataset_config.ResultFolder
        response = {**run_results}
        try:

            timezone = pytz.timezone("America/Los_Angeles")
            created_timestamp = datetime.fromtimestamp(time.time(), tz=timezone)
            created_timestamp_str = created_timestamp.strftime("%Y-%m-%d_%H:%M:%S_%Z")

            email_recipients = ["yangj@nvidia.com"]
            # logger.info(f"email recipients: {email_recipients}")
            if email_recipients:
                email_service_response = EmailServices().send_email(
                    f"Regression reports - {self.project}",
                    email_recipients,
                    {
                        "metrics": {"Processed": self.metrics.Processed},
                        "project_name": f"{self.project} on {self.env} environment created at {created_timestamp_str}",
                        "run_status": run_results.get("status") or "exception",
                        "details": run_results,
                    },
                    "eval-run-report.html",
                )
                response = {**response, **email_service_response}

        except Exception as ex:
            response["status"] = "exception"
            response["exception"] = f"{ex}"

        return response

    # TODO: config -> object
    @log_errors("Prepare Regression Run")
    async def prepare(
        self,
        request: RunMakerRequest,
        config: NVBotEvaluationConfig,
        project_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[DataFrame]:
        # run_config = config.get("RunConfig", None)
        run_config = (
            config.RegressionSchema.RunConfig
            if is_valid_instance(config.RegressionSchema)
            else None
        )
        if run_config:
            RegressionRunContainer.input_mapper = await dict_parser(
                parser_config=run_config.dict().get("Inputs")
            )
            RegressionRunContainer.output_mapper = await dict_parser(
                parser_config=run_config.dict().get("Outputs")
            )

        dataset_config = config.RegressionSchema.DatasetConfig
        # dataset_config = config.get("DatasetConfig", None)
        logger.info(f"Dataset config: {dataset_config.model_dump_json()}")
        if is_valid_instance(dataset_config):
            # dataset_folder = dataset_config.DatasetFolder or self.project.lower()
            data_limit = dataset_config.DataLimit
            data_shuffle_seed = dataset_config.DataShuffleSeed
            dataset_filters = dataset_config.Filters

            df = load_data_from_datasetconfig(self.project, dataset_config)
            return df

        return None

    @log_errors("Generate local excel output name")
    def _generate_excel_filepath(self, request: RunMakerRequest):
        str_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
        directory = LOCAL_EVAL_RESULTS_TMP_FOLDER
        create_folder_if_not_exists(directory)
        file_path = (
            f"{LOCAL_EVAL_RESULTS_TMP_FOLDER}/{request.Project}-{str_datetime}.xlsx"
        )
        print(f"Regression local file path: {file_path}")
        return file_path


@log_errors("Save evaluation results")
def _save_evaluation_history_to_database(
    request: RunMakerRequest, status: EvaluationRunStatus
) -> None:
    try:
        evaluation_history_id = DatabaseHandler(
            {"env": "dev"}
        ).add_to_evaluation_history(
            run_request=request,
            dataset_metadata={},
            evaluation_metadata=(
                request.EvaluationConfig.model_dump(exclude_unset=True)
                if request.EvaluationConfig
                else {}
            ),
            flowconfig_metadata=(
                request.PlatformConfig.model_dump(exclude_unset=True)
                if request.PlatformConfig
                else {}
            ),
            output_url="",
            status=str(status.value),
        )
        # print(f"Saved evaluation history in database: {evaluation_history_id}")
        logger.info(f"Saved evaluation history in database: {evaluation_history_id}")

        # verify
        response = DatabaseHandler({"env": "dev"}).get_evaluation_history(
            history_id=int(evaluation_history_id),
        )
        return response

    except Exception as ex:
        print(f"Exception in save evaluation history to database, {ex}")
        raise Exception(ex)
