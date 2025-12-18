from __future__ import annotations

import asyncio

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytz

from pandas import DataFrame
from tqdm import tqdm
from configs.settings import get_cache_session, get_chatbot_url

from data_models.api.evaluation_result import RegressionRunMetrics
from data_models.dataset_handler import DataContentOutputConfig
from data_models.api.run_maker import RunMakerRequest, NVBotEvaluationConfig, StorageType, FlowConfigRequest
# from nv_platform.nvbot_platform import NVBotPlatform
from service_library.constants import LOCAL_EVAL_RESULTS_TMP_FOLDER
from service_library.handler.config_loader import ConfigLoader
from service_library.handler.s3_dataset_handler import S3DatasetHandler
from service_library.nemo_ms.nemo_service_helper import get_extract_run_maker_request
from service_library.notification.send_email import EmailServices
from service_library.run.parser_library.input_parser import format_chat_request_post_params
from service_library.run.parser_library.response_parser import fetch_post_chatbot_request_response
from service_library.run.parser_library.dict_parser import dict_parser
from service_library.run.run_container import RunContainer
from service_library.utils.data_helper import file_content_convertor
from service_library.utils.generalize_response_helper import generalize_response, create_query_variation
from service_library.utils.logging import log_errors
from service_library.utils.request_helpers import create_header
from service_library.utils.run_helpers import generate_excel_output, \
    create_folder_if_not_exists, convert_query_to_snake_case

# from utility import generate_regression_filename

logger = logging.getLogger(__name__)


class CacheUpsertContainer(RunContainer):
    """A container to help manage the state of an eval run."""

    input_mapper = None
    output_mapper = None

    def __init__(self, project: str, config: NVBotEvaluationConfig, env: str):
        super().__init__(project, config, env)
        self.chatbot_url = get_chatbot_url(env)  # default

        self.input_mapper = None
        self.output_mapper = None

        self.metrics = RegressionRunMetrics()
        self.metrics_name = ""

        self.temp_filepath = None

    @log_errors('Answer generation')
    def insert_to_cache(self, collection_key, query_key, value, expire_after=60 * 60 * 24 * 70):

        cache_session = get_cache_session()
        query_key_hash = f'{collection_key}_response_cache'.upper()

        print(f"The cache server is: {cache_session.redis_url}")
        print(f"The collection key hash is: {query_key_hash}")
        print(f"The key to be inserted is: {query_key}")

        return cache_session.update_expire_cache(query_key,
                                                 value,
                                                 expire_after,  # seconds
                                                 query_key_hash)

    @log_errors('Answer generation')
    async def arun(
            self,
            request: RunMakerRequest,
            config: NVBotEvaluationConfig,
            df: DataFrame,
            **kwargs):

        print(f"CacheUpdate arun {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}")
        headers = create_header(self.auth_token)

        results = []
        self.metrics.Total = df.shape[0]
        self.temp_filepath = self._generate_excel_filepath(request)

        bot_name = request.System
        if bot_name.lower().startswith("nvinfo"):
            bot_name = "nvinfo"
        # async with aiohttp.ClientSession(timeout=timeout) as request_session:
        if self.chatbot_url:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                time.sleep(1)
                try:
                    start = time.time()

                    print(f"\n# {index} start: {datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')}")
                    # TODO: build request parameters (prepare_chatbot_request)
                    row_dict = row.to_dict()

                    processed_result = {**row}

                    defined_generalize_response = row.get("Generalize response")
                    # response = processed_result.get("Response", "")
                    if defined_generalize_response:
                        print(f'⚠️#{index} - {row.get("Query")} response will use existing suggested response')
                        processed_result["Generalize response"] = defined_generalize_response
                        parsed_response = processed_result
                    else:

                        chat_request_params = await format_chat_request_post_params(request, row["Query"])

                        self.chatbot_url = chat_request_params.get("chatbot_url")
                        chatbot_request = chat_request_params.get("chatbot_request")
                        chatbot_request.Query = row["Query"]

                        result = {"Query": chatbot_request.Query}
                        chat_request_params = await format_chat_request_post_params(
                            get_extract_run_maker_request(request), "")

                        self.chatbot_url = chat_request_params.get("chatbot_url")
                        chatbot_request = chat_request_params.get("chatbot_request")
                        env = request.Env
                        result_dict = {}
                        latencies = []
                        print("Platform run at --> ", self.chatbot_url)

                        bot_config_request = FlowConfigRequest.model_validate(request)
                        bot_config = await ConfigLoader(self.env).get_bot_config(bot_config_request)

                        if bot_config:
                            bot_name = bot_config.get("botName")

                            if bot_name.lower().startswith("nvinfo"):
                                bot_name = "nvinfo"

                        if row["Query"]:
                            parsed_response, retry_count, duration_avg = await fetch_post_chatbot_request_response(chatbot_request, self.chatbot_url,
                                                                                  env)

                        assert parsed_response is not None, f"Failed to get chatbot response {chatbot_request.Query}"
                        print(f"#{index + 1} Response status: {parsed_response.get('status')}")
                        logging.info(f"#{index + 1} Response status: {parsed_response.get('status')}")
                        if parsed_response.get("response"):
                            chatbot_response = parsed_response.get("response")
                            assert chatbot_response, f" #{index + 1} - Failed to get response"
                            # if chatbot_response is not None:
                            # for direct response
                            if isinstance(chatbot_response, str):
                                result["Response"] = chatbot_response
                                processed_result = result
                            # for structured response
                            elif isinstance(chatbot_response, list):
                                parser_result = CacheUpsertContainer.output_mapper(chatbot_response)
                                processed_result = parser_result

                            elif isinstance(chatbot_response, dict):
                                processed_result = CacheUpsertContainer.output_mapper({**chatbot_response, **result})
                                # processed_result = {**result, **parser_result}
                            end = time.time()
                            print("end: ", datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S'))
                            latencies.append(end - start)

                            processed_result["Generalize response"] = generalize_response(row['Query'],
                                                                                          processed_result.get(
                                                                                              "Response"))

                            processed_result["Duration"] = round(end - start, 2)
                            # processed_result["IsStream"] = str(chatbot_request.Parameters.IsStream)
                            # processed_result["Request"] = json.dumps(chatbot_request.dict())
                            # pprint.pprint(chatbot_request.dict())

                    query_key = row.get("Query Key")
                    query_variation = row.get("Query Variation")

                    if not query_key:
                        query = convert_query_to_snake_case(row["Query"])
                        query_key = f"{bot_name.upper()}_QUERY_CACHE_KEY:{query}"
                    if not query_variation:
                        query_variation = create_query_variation(row['Query'], processed_result["Generalize response"])
                        if isinstance(query_variation, list):
                            query_variation = "\n".join(query_variation)

                    should_upsert = request.RunType.lower() == "upsert"
                    if should_upsert and self.insert_to_cache(bot_name, query_key,
                                                              processed_result["Generalize response"]):
                        print(f"Successfully inserted {row['Query']} to cache")
                        logging.info(f"Successfully inserted {row['Query']} to cache")
                        self.metrics.Processed += 1
                        processed_result["Upsert Status"] = "complete"
                    elif should_upsert:
                        print(f"Failed inserted {row['Query']} to cache")
                        logging.warning(f"Failed inserted {row['Query']} to cache")

                    processed_result["Query Variation"] = query_variation
                    results.append(processed_result)

                    if processed_result:
                        for key, value in processed_result.items():
                            if isinstance(value, (str, float, int, np.ScalarType)):
                                df.loc[index, key] = value
                            elif isinstance(value, (dict, list)):
                                df.loc[index, key] = str(value)
                            # result = RegressionRunContainer.output_mapper(resp)
                    else:
                        error_message = f"🚨Failed to get response from {self.chatbot_url}, details: {parsed_response}"
                        logging.error(error_message)
                        print(f"\n{error_message}\n")


                except asyncio.TimeoutError:
                    print(f"The request timed out")
                except Exception as ex:
                    message = f"Exception in regression run on #{index} Query {row['Query']}: {ex}"
                    print(message)
                    raise Exception(message)
            # assert len(
            #     latencies) > 0, f"🚨Errors received on create chatbot response from {self.chatbot_url}, please validate request params."

            # self.metrics.Avg = round(np.average(latencies), 2)
            # self.metrics.P50 = round(np.percentile(latencies, 50), 2)
            # self.metrics.P95 = round(np.percentile(latencies, 95), 2)
            # self.metrics.P99 = round(np.percentile(latencies, 99), 2)
            # self.metrics.Std = round(np.std(latencies, ), 2)
            print(f"Cache Update arun completes: {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}")
            print("Cache Update: ", self.metrics.dict(exclude_none=True))

            query_variation_dict = {}
            for r in results:
                query_variation_dict[r.get("Query Key")] = r.get("Query Variation")

            generate_excel_output(self.temp_filepath, ['Results'], [df], ["Query variation"],
                                  query_variation_dict={"query_variation": query_variation_dict})

            # print ("\n-------query_variation_dict-------\n")
            # print(query_variation_dict)
            # print("\n------query_variation_dict--------\n")

        return {
            "local_filepath": self.temp_filepath,
            "results": results
        }

    @log_errors("Finish run")
    def finish(self, request: RunMakerRequest, config: NVBotEvaluationConfig, run_results: dict = {}):
        notification = config.RegressionSchema.Notification
        response = {**run_results}
        try:
            timezone = pytz.timezone('America/Los_Angeles')
            created_timestamp = datetime.fromtimestamp(time.time(), tz=timezone)
            created_timestamp_str = created_timestamp.strftime('%Y-%m-%d_%H:%M:%S_%Z')

            email_recipients = notification.EmailRecipients
            # logger.info(f"email recipients: {email_recipients}")
            if email_recipients and len(email_recipients) > 0:
                icon = "🏖" if not request.RunType.lower() == "upsert" else "✍️"
                email_service_response = EmailServices().send_email(f"{icon}Cache upsert - {self.project}",
                                                                    email_recipients,
                                                                    {
                                                                        "metrics": {"Processed": self.metrics.Processed,
                                                                                    "Total": self.metrics.Total},
                                                                        "project_name": f"{self.project} cache upsert at {created_timestamp_str} towards {get_cache_session().redis_url}",
                                                                        "run_status": run_results.get(
                                                                            "status"),
                                                                        "creation_time": created_timestamp_str,
                                                                    }
                                                                    )
                response = {**response, **email_service_response}

        except Exception as ex:
            response["status"] = "exception"
            response["exception"] = f"{ex}"

        return response

    # TODO: config -> object
    @log_errors("Prepare Cache Update Run")
    async def prepare(
            self,
            request: RunMakerRequest,
            config: NVBotEvaluationConfig,
            project_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[DataFrame]:
        # run_config = config.get("RunConfig", None)
        run_config = config.RegressionSchema.RunConfig if config.RegressionSchema else None
        if run_config:
            CacheUpsertContainer.input_mapper = await dict_parser(parser_config=run_config.dict().get("Inputs"))
            CacheUpsertContainer.output_mapper = await dict_parser(parser_config=run_config.dict().get("Outputs"))

        dataset_config = config.RegressionSchema.DatasetConfig
        # dataset_config = config.get("DatasetConfig", None)
        logging.info("Dataset config:", dataset_config)
        if dataset_config:
            # dataset_folder = dataset_config.DatasetFolder or self.project.lower()
            data_limit = dataset_config.DataLimit
            data_shuffle_seed = dataset_config.DataShuffleSeed

            df = None
            print(f"Dataset config engine: {dataset_config.Engine.lower()}")
            if dataset_config.Engine.lower() == StorageType.S3.value:
                df = S3DatasetHandler(
                    self.project,
                    dataset_config.dict()
                ).download_file(
                    folder_name=dataset_config.Name,
                    file_name=dataset_config.DatasetPath,
                    output_config=DataContentOutputConfig(
                        Format="pd"))

            elif dataset_config.Engine.lower() == StorageType.DATASTORE.value:
                # TODO: add read from datastore
                pass
            else:
                # HACK: if want to use local file, can use this to read local files
                file = dataset_config.RunFile
                df = file_content_convertor("pd", file)

                # df = pd.read_csv('script/avc_eval_dataset.csv', encoding='utf-8')
            if df is None or isinstance(df, dict):
                logging.error(f"Failed loading data from: {dataset_config.Engine.lower()}")
                print(f"no dataset found from: {dataset_config.Engine.lower()}")
            if df is not None and data_limit is not None and df.shape[0] > data_limit:
                df = df[0:data_limit]
            if df is not None and data_shuffle_seed is not None:
                df = df.sample(frac=1, random_state=abs(data_shuffle_seed))
            df = df.where(pd.notna(df), None)
            return df

        return

    # def run_without_inference(self, request: RunMakerRequest, df: DataFrame):
    #     self.temp_filepath = self._generate_excel_filepath(request)
    #     for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    #         try:
    #             self.metrics.Processed += 1
    #             start = time.time()
    #             # TODO: build request parameters (prepare_chatbot_request)
    #             row_dict = row.to_dict()
    #             request_data = RegressionRunContainer.input_mapper(row_dict)
    #
    #         except Exception as ex:
    #             print(f"Exception in regression run_without_inference, {ex}")
    #             raise Exception(ex)
    #
    #     file_path = generate_excel_output(df, self.temp_filepath, self.metrics.dict())
    #     logger.info(f"Result saved to {file_path}")
    #     return self.temp_filepath

    # @log_errors("Generate local excel output name")
    def _generate_excel_filepath(self, request: RunMakerRequest):
        str_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
        directory = LOCAL_EVAL_RESULTS_TMP_FOLDER
        create_folder_if_not_exists(directory)
        file_path = f"{LOCAL_EVAL_RESULTS_TMP_FOLDER}/{request.Project}-{str_datetime}.xlsx"
        print(f"Cache update local file path: {file_path}")
        logger.info(f"Cache update result saved at: {file_path}")
        return file_path
