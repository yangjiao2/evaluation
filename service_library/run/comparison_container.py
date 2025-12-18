import logging
import os
import re
from urllib.parse import urlparse, parse_qs

from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Callable, Tuple
import pandas as pd
import pytz
from pandas import DataFrame
from scipy.stats import stats
from nvbot_sdk.chat_llm_wrappers.nvcf_chat_completions import NVCFChat
from tqdm import tqdm

from data_models.api.run_maker import ComparisonSchema, DatasetConfig, StorageType, NVBotEvaluationConfig, \
    RunMakerRequest, FlowConfigRequest
import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score
import json
# from sklearn.ensemble import IsolationForest
from datetime import datetime

from eval_prompt_library.consistency_eval_prompt import ConsistencyEvaluationPrompt
from nvbot_models.request_models.evaluation_request import UserEvaluationRunData, EvaluationRunStatus
from service_library.constants import LOCAL_EVAL_RESULTS_TMP_FOLDER, NT_ACCOUNT_ID, NT_ACCOUNT_NAME
from service_library.handler.database_handler import DatabaseHandler
from service_library.nemo_ms.nemo_service_helper import create_dataset_and_upload_folder
from service_library.notification.send_email import EmailServices
from service_library.run.run_container import RunContainer
from service_library.utils.data_helper import calculate_trend, is_increasing, is_decreasing
from service_library.utils.datasetconfig_helper import load_data_from_datasetconfig, is_same_datasetconfig
from service_library.utils.logging import log_errors
from configs.settings import get_settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from collections import Counter

from service_library.utils.pydantic_helper import is_valid_instance
from service_library.utils.run_helpers import generate_excel_output, get_formatted_datetime, generate_short_uuid, \
    create_folder_if_not_exists

COLUMN_COMPARISON_SEPARATOR = ":"
COLUMN_SUFFIX_SEPARATOR = "_"

logger = logging.getLogger(__name__)

settings = get_settings()

model = NVCFChat(redis_secrets=settings.REDIS_CACHE_SECRET,
                 nvcf_api_key=settings.PUBLIC_NVCF_API_KEY,
                 model_name="meta/llama-3.1-70b-instruct",
                 model_url="https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/d639db7e-fa82-4e24-9a53-cab7631f4804",
                 max_tokens=1500, temperature=0.0)


class ComparisonContainer(RunContainer):
    def __init__(self, project: str, config: NVBotEvaluationConfig, env: str = ""):
        super().__init__(project, config, env)
        self.input_mapper = None
        self.output_mapper = None

        self.metrics = {}
        self.temp_folderpath = None

        self.EVAL_URL = self.settings.NEMO_EVAL_URL
        self.DS_URL = self.settings.NEMO_DS_URL

        self.datasets = {}  # eval_id -> data
        self.comparison_group = {}
        self.scorers = []

        self.customEvaluator = config.ComparisonSchema.Comparator if is_valid_instance(config.ComparisonSchema) else {}

    @log_errors("Initiate")
    def initiate(self, request: RunMakerRequest):

        # pre-process update dataset
        dataset_configs = []

        for dataset_config in self.config.ComparisonSchema.DatasetConfigs:
            if dataset_config.Engine.lower() == "database":
                if dataset_config.HistoryId:
                    history_datasetconfig = DatabaseHandler({'env': 'dev'}).get_detail_result_as_datasetconfig(
                        history_id=dataset_config.HistoryId)
                    if history_datasetconfig:
                        dataset_configs.append(history_datasetconfig)

        self.config.ComparisonSchema.DatasetConfigs = dataset_configs
        # response = _save_evaluation_history_to_database(request, {"DatasetConfigs": dataset_configs}, {})

        return {"id": None}  # dataset_configs

    @log_errors("Build DatasetConfigs")
    def build_datasetconfigs(self, current_history: dict) -> NVBotEvaluationConfig:
        if not is_valid_instance(self.config.ComparisonSchema):
            return None
        if self.config.ComparisonSchema.DatasetConfigs and len(self.config.ComparisonSchema.DatasetConfigs) > 0:
            return self.config

        history_filter = {
            "project": current_history.get("project"),
            "project_id": current_history.get("project_id"),
            "run_type": current_history.get("run_type"),
            "tag1": current_history.get("tag1"),
            "tag2": current_history.get("tag2"),
            "status": "COMPLETED",
            "created_date_from": get_formatted_datetime(None, {"days": -30})
        }

        # update datasetconfig to format from DB
        total = 2
        dataset_configs = []
        evaluation_schema = NVBotEvaluationConfig.model_validate(current_history['evaluation_metadatajson'])
        evaluation_history_metadata = json.loads(current_history.get("metadata_value"))
        env = FlowConfigRequest.model_validate(evaluation_history_metadata).Env

        page_number = 0
        while len(dataset_configs) < total:
            page_number += 1
            previous_runs = DatabaseHandler({'env': 'dev'}).get_evaluation_history_records(
                filters=history_filter,  # add status
                pagination={
                    "page": page_number,
                    "size": 100
                }
            ).get("items")
            previous_runs = [item for item in previous_runs if item["id"] < current_history['id']]
            if not previous_runs:
                break
            for h in previous_runs:
                if len(dataset_configs) > total:
                    break
                if (datetime.fromisoformat(current_history.get("created_date")) - datetime.fromisoformat(
                        h.get("created_date"))).days > 30:
                    break
                # check env, tags
                h_evaluation_history_metadata = json.loads(h["metadata_value"])
                h_env = FlowConfigRequest.model_validate(h_evaluation_history_metadata).Env
                if h_env.lower() != env.lower() or h.get("tag1", "") != current_history.get("tag1", "") \
                        or h.get("tag2", "") != current_history.get("tag2", "") \
                        or h.get("tag3", "") != current_history.get("tag3", ""):
                    continue

                # check dataset config
                h_evaluation_schema = NVBotEvaluationConfig.model_validate(h['evaluation_metadatajson'])
                if not is_same_datasetconfig(evaluation_schema.RegressionSchema.DatasetConfig,
                                             h_evaluation_schema.RegressionSchema.DatasetConfig):
                    continue

                history_datasetconfig = DatabaseHandler({'env': 'dev'}).get_detail_result_as_datasetconfig(
                    history_id=str(h['id']))
                if history_datasetconfig:
                    dataset_configs.append(history_datasetconfig)

        if len(dataset_configs) >= total:
            self.config.ComparisonSchema.DatasetConfigs = dataset_configs[-2:]
            return self.config
        return None

    @log_errors('Trigger comparison Eval')
    async def arun(
            self,
            request: RunMakerRequest,
            config: NVBotEvaluationConfig,
            df: DataFrame,
            **kwargs):
        combined_data, comparison_results = self._compare_datasets(request, df)
        trends_metrics = None
        format_checks = None
        reports = comparison_results.copy()

        dataconfig = config.ComparisonSchema.DataConfigs
        column_map = dataconfig.get("column_map")
        query_column = column_map.get("question")
        category_column = column_map.get("category")

        if is_valid_instance(self.config.ComparisonSchema.Outlier):
            combined_data = await self.classify_outliers(request, config, combined_data)

        trends_metrics = calculate_trend(combined_data, self.scorers, category_column)
        reports["Trend Metrics"] = trends_metrics.to_dict()
        return {
            "results_df": combined_data,
            "trends_metrics_df": trends_metrics,
            "reports": reports
        }

    async def classify_outliers(self,
                                request: RunMakerRequest,
                                config: NVBotEvaluationConfig,
                                combined_data: DataFrame):
        """
        Classifies outlier responses using an LLM and logs the results.

        Args:
            combined_data (pd.DataFrame): The dataset containing responses.
            model: The LLM model instance.
            config: Configuration object containing schema mappings.
            comparison_group (dict): Mapping for response groups.
            scorers (list): List of scorer categories.

        Returns:
            pd.DataFrame: Updated DataFrame with LLM-detected outliers.
        """
        runnable = ConsistencyEvaluationPrompt | model
        dataconfig = config.ComparisonSchema.DataConfigs
        column_map = dataconfig.get("column_map")
        query_column = column_map.get("question")
        category_column = column_map.get("category")

        reports = []

        for index, row in tqdm(combined_data.iterrows(), total=len(combined_data)):
            try:
                # Extract outlier summary
                outlier_summary = row.get("Outlier Summary")

                # Prepare response scores
                response_scores = {
                    f"Response.{i + 1}": [
                        f"{res_score}: {row.get(self.comparison_group[res_score][i], 'N/A')}"
                        for res_score in self.scorers
                    ]
                    for i, res in enumerate(row[self.comparison_group["Response"]])
                }

                # Construct input for LLM
                inputs = {
                    'scorers': self.scorers,
                    'question': row.get("Query"),
                    'reference': row.get("Correct Answer"),
                    'required_citation': row.get("Required Citation"),
                    'responses': '\n'.join([
                        f"{key}: {text}" for key, text in row[self.comparison_group["Response"]].items()
                    ]),
                    'response_scores': json.dumps({k: '; '.join(v) for k, v in response_scores.items()})
                }

                # Invoke the LLM
                consistency_output = await runnable.ainvoke(inputs)
                llm_output = json.loads(consistency_output.content)

                # Extract LLM outlier classification
                llm_outlier = llm_output.get("Outlier", "").lower()
                explanation = llm_output.get("Explanation", "")

                if llm_outlier and llm_outlier != "none":
                    combined_data.loc[index, 'Outlier Summary (LLM)'] = llm_outlier
                    combined_data.loc[index, 'Outlier Explanation (LLM)'] = explanation

                    reports.append(
                        f"[#{index}] Query: {row.get(query_column, '')}\n"
                        f"- Outlier Response: {llm_outlier} ({row.get(llm_outlier, '')})"
                    )

            except Exception as ex:
                print(f"Error processing row {index}: {ex}")

        return combined_data

    def get_outlier_row_index(self, data: pd.DataFrame):

        # Find columns that start with "Outlier Summary"
        outlier_columns = [col for col in data.columns if col.startswith("Outlier Summary")]

        # Find row indices where any of these columns have non-empty, non-NaN, non-None values
        indices = []
        for index, row in data.iterrows():
            for col in outlier_columns:
                if row[col] not in [None, np.nan, '']:
                    indices.append(index)
                    break  # Stop checking other columns for this row if a non-empty value is found

        return indices

    @log_errors("Finish run")
    def finish(self, request: RunMakerRequest, config: NVBotEvaluationConfig, run_results: dict,
               evaluation_history: dict):
        response = {"status": "success"}

        comparison_results = run_results.get("results_df")
        trends_metrics = run_results.get("trends_metrics_df").to_dict()
        reports = run_results.get("reports")

        directory = LOCAL_EVAL_RESULTS_TMP_FOLDER

        def _construct_result_filename(evaluation_history, datetime):
            # e.g: bot_config_name-env-datetime-metric_number
            joint_dataset_names = "-".join(list(self.datasets.keys()))

            return f"{joint_dataset_names}_{datetime}"

        def _construct_created_timestamp_str_from_created_date(created_time_datetime: str, verbose: bool = False):
            utc_time = datetime.strptime(created_time_datetime, "%Y-%m-%dT%H:%M:%S.%f")

            # Assign the UTC timezone to the datetime object
            utc_time = utc_time.replace(tzinfo=pytz.utc)

            # Convert the datetime object to US/Pacific time
            pacific_tz = pytz.timezone('US/Pacific')
            created_timestamp = utc_time.astimezone(pacific_tz)
            created_timestamp_str = created_timestamp.strftime('%Y-%m-%d_%H:%M:%S_%Z' if verbose else '%Y%m%d-%H%M%S')
            return created_timestamp_str

        # created_timestamp_str = _construct_created_timestamp_str_from_created_date(
        #     created_time_datetime=evaluation_history.get('created_date',
        #                                                  datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"))
        # )

        result_filename = _construct_result_filename(evaluation_history, self.project or "")

        dataset_name = f"comparison-{evaluation_history.get('id') or result_filename}-{generate_short_uuid(5)}"
        create_folder_if_not_exists(os.path.join(str(directory), str(dataset_name)))

        output_file = os.path.join(directory,
                                   dataset_name,
                                   result_filename)

        file_path = generate_excel_output(
            output_file,
            [{'Consistency': {
                "index": self.get_outlier_row_index(comparison_results),
                "color": "#ffd966"
            }}],
            [run_results.get("results_df")],
            list( reports.keys() ),
            **reports
        )
        print(f"📁File path: {file_path}")
        # self.outliers = self._generate_outliers(self.results)

        # TODO: upload

        dir_name, base_name = os.path.dirname(file_path), os.path.basename(file_path)
        upload_result_id, path = create_dataset_and_upload_folder(self.DS_URL, dataset_name, dir_name, path_in_repo=".")
        print("upload answers response: ", upload_result_id)
        logger.info(f"Processed comparison results uploaded to datastore: {upload_result_id}")

        # ds_content = DatasetConfig(
        #     Engine=StorageType.DATASTORE.value,
        #     DatasetId=upload_result_id,
        #     Files=[base_name],
        #     RunFile=file_path
        # )

        file_path_wrapped_by_braces = "{filepath}"
        result_detail_url = f"{self.settings.NVBOT_EVALUATION_URL}/datasets/download/{upload_result_id}/file/contents/{file_path_wrapped_by_braces}?file_path={base_name}"
        # send email
        notification_data = {
            # "bot_name": bot_name,
            "run_id": evaluation_history.get("id", ""),
            "run_status": "succeeded",
            "run_info": f'comparison result between {result_filename}',
            "metrics": trends_metrics,
            "project_name": f"{self.project} environment created at {_construct_created_timestamp_str()}",
            "result_url": result_detail_url,
            # "env": self.env,
            # "creation_time": created_timestamp_str,
            "tags":
                list(filter(lambda x: x is not None, [evaluation_history.get("tag1"), evaluation_history.get("tag2"),
                                                      evaluation_history.get("tag3")])),
            "run_url": f"{get_settings().NVBOT_EVALUATION_UI_URL}/dashboard/analytics?id={evaluation_history.get('id')}",
            "report": """   Comparison result \n  
            - 5555   """
        }

        response.update({"result_detail_url": result_detail_url, "history_id": evaluation_history.get("id", ""),
                         "metrics": trends_metrics})

        email_result = self.send_notification(notification_data)
        if email_result:
            response.update(email_result)
        if evaluation_history:
            try:
                history_id = evaluation_history.get("id")
                param = UserEvaluationRunData(
                    Id=history_id,
                    Project=self.project,
                    NtAccount=NT_ACCOUNT_ID,
                    Username=NT_ACCOUNT_NAME,
                    Status=EvaluationRunStatus.COMPLETED.value
                )
                update_response = DatabaseHandler({'env': 'dev'}).update_evaluation_history_and_details(param)
                assert update_response, f"Failed to update database for history and details info: {update_response}"
                response["history_id"] = update_response.get("id")

            except Exception as ex:
                logger.error(f"Failed to update database for history and details info: {ex}")

        return response

    @log_errors("Send Notification")
    def send_notification(self, notification_data: dict):
        allow_notification = self.config.ComparisonSchema.Notification

        if allow_notification:
            try:
                email_recipients = allow_notification.EmailRecipients
                logger.info(f"email recipients: {email_recipients}")
                if email_recipients:
                    status_icon = "🔍" if notification_data["run_status"].lower() == "succeeded" \
                        else "⚠️" if notification_data["run_status"].lower() == "failed" else "🤔"
                    email_service_response = EmailServices().send_email(
                        f"{status_icon} consistency reports - {notification_data['project_name']}",
                        email_recipients,
                        notification_data
                    )
                    logger.info(f"email_service_response: {email_service_response}")
                    return {"notification": email_service_response}
            except Exception as ex:
                logger.error(f"error when sending email notification: {ex}")

    @log_errors("Prepare Comparison Eval")
    async def prepare(
            self,
            request: RunMakerRequest,
            config: NVBotEvaluationConfig,
            project_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[DataFrame]:

        """
        Load datasets based on DatasetConfigs provided in the comparison_schema.
        """

        data_configs = config.ComparisonSchema.DataConfigs
        column_name_mapping = data_configs.get("column_map", {})
        self.scorers = column_name_mapping.get("scorers")

        result = []
        # comparison_columns = NVINFO_COLUMN_NAMES  # config.ComparisonSchema.DataConfig.get("columns")
        for index, dataset_config in enumerate(config.ComparisonSchema.DatasetConfigs):
            print(f"Loading from {dataset_config.Engine.lower()}")
            history_id = dataset_config.HistoryId
            dataset_key = f"{dataset_config.Engine.lower()}{index}"
            if history_id:
                dataset_key = history_id
            if history_id:
                history = DatabaseHandler({'env': 'dev'}).get_evaluation_history(
                    history_id=int(history_id),
                )

                nemo_eval_id = json.loads(history.get("output_url", {})).get("id", "")
                # if nemo_eval_id:
                #     dataset_key = history.get("nemo_eval_id")

            df = load_data_from_datasetconfig(self.project, dataset_config)

            if df is not None:
                self.datasets[dataset_key] = df

        # trim df
        self.datasets = _check_and_trim_dfs(self.datasets)

        ordered_columns = []
        for index, (df_key, df) in enumerate(self.datasets.items()):
            column_names = list(df.columns)
            before_response, after_response = _divide_columns_by_response(column_names)
            if index == 0:
                # common_columns = [col for col in df.columns if col not in comparison_columns]
                result = df[before_response]
                ordered_columns = before_response

            if after_response:
                # Filter only the columns present in after_response
                columns = [col for col in after_response if col in df.columns]
                df = df[columns]

                # Create a mapping of column groups to their renamed versions
                grouped_columns = {}
                for column in df.columns:
                    col_label = f"{column}.{index + 1}"
                    if column not in self.comparison_group:
                        self.comparison_group[column] = []
                    self.comparison_group[column].append(col_label)
                    grouped_columns[column] = col_label

                # Rename columns in the current DataFrame
                df = df.rename(columns=grouped_columns)

                # Add renamed columns to result
                result = pd.concat([result, df], axis=1)

        # Append the suffixed columns grouped by their base name
        for base_column in self.comparison_group:
            ordered_columns.extend(self.comparison_group[base_column])

        # Reorder the DataFrame columns
        result = result[ordered_columns]

        return result
        # if True:
        #     if after_response:
        #         # assert "Response" in df.columns, "Expecting to have response"
        #         columns = [col for col in after_response if col in df.columns]
        #         df = df[columns]
        #
        #         for column in df.columns:
        #             col_label = f"{column}.{index + 1}"
        #             if column not in self.comparison_group:
        #                 self.comparison_group[column] = []
        #             self.comparison_group[column].append(col_label)
        #             df = df.rename(columns={column: col_label})
        #
        #         # identify_column = pd.DataFrame([None] * len(result), columns=[df_key or f"{index + 1}"])
        #         result = pd.concat([result, df], axis=1)

    @log_errors("Compare Dataset")
    def _compare_datasets(self, request: RunMakerRequest, combined_data: pd.DataFrame) -> Tuple[
        pd.DataFrame, Dict[str, Any]]:
        """
        Compare datasets by calculating similarities between columns with the same prefix.
        """
        comparison_results = {}
        if is_valid_instance(self.config.ComparisonSchema.Comparator):
            for op in self.config.ComparisonSchema.Comparator:
                column_name = op.get("column_name")

                comparison_columns = self.comparison_group[column_name]
                print(f'columns to be compared: {comparison_columns}')
                op_method = op.get("method").lower()

                # Step 1: Pairwise comparison
                for i in range(len(comparison_columns)):
                    for j in range(i + 1, len(comparison_columns)):
                        col1 = comparison_columns[i]
                        col2 = comparison_columns[j]
                        new_col_name = f"{col1}:{col2}"
                        if op_method == "tfidf":
                            # Add TF-IDF similarity score
                            new_col_name_tfidf = f"{col1}:{col2}({op_method})"
                            combined_data = add_similarity_score(combined_data, col1, col2, apply_tfidf_similarity,
                                                                 new_col_name_tfidf)
                        elif op_method == "bert":
                            new_col_name_bert = f"{col1}:{col2}({op_method})"
                            combined_data = add_similarity_score(combined_data, col1, col2, apply_bert_f1,
                                                                 new_col_name_bert)

                # Step 2: Detect outliers based on similarity scores and thresholds
                threshold = op.get("threshold", 0.7)

                # Define score_column_names based on BERT similarity by default
                score_column_names = [f"{comparison_columns[i]}:{comparison_columns[j]}({op_method})"
                                      for i in range(len(comparison_columns))
                                      for j in range(i + 1, len(comparison_columns))]

                def _find_outliers(row, score_column_names, threshold):
                    """
                    Identifies outliers in a row based on the provided threshold for each score column.
                    Appends the outlier column names where the score is greater than the threshold.

                    :param row: A row of the DataFrame.
                    :param score_column_names: A list of score column names to check for outliers.
                    :param threshold: The threshold value to identify outliers.
                    :return: A comma-separated string of outlier column names.
                    """
                    # Find columns where the value is greater than the threshold
                    print(row['ID'], score_column_names)
                    outliers = [col for col in score_column_names if row[col] < threshold]

                    # print('outliers:', outliers)
                    # Return the list of outlier column names as a comma-separated string
                    return ", ".join(outliers)

                # Apply the outlier detection to each row in the DataFrame
                column_name_pair = f"{column_name} Divergent Pair"
                print("outliers: ", column_name_pair)
                # print(score_column_names)
                combined_data[column_name_pair] = combined_data.apply(
                    lambda row: _find_outliers(row, score_column_names, threshold), axis=1
                )

                # Step 5: Find the most frequent outlier for each row
                def most_frequent_outlier_for_row(row, outlier_column: str) -> str:
                    """
                    Find the most frequent outlier(s) for a given row based on the specified outlier column.

                    :param row: A row from the DataFrame.
                    :param outlier_column: The name of the column that contains the outlier information.
                    :return: A string with the most frequent outlier column(s).
                    """
                    outlier_pairs = row[outlier_column].split(", ")
                    if not outlier_pairs or outlier_pairs == ['']:
                        return ''

                    most_frequent_outliers = ""
                    outliers = []
                    try:
                        # Extract individual outlier columns
                        for outlier in outlier_pairs:
                            # Remove the "(bert)" part and split by colon
                            cols = re.sub(r'\(.*?\)', '', outlier).split(':')
                            outliers.extend(cols)

                        # Count the frequency of each individual outlier
                        outlier_counter = Counter(outliers)

                        # Find the most frequent outlier(s)
                        max_frequency = max(outlier_counter.values())
                        most_frequent_outliers = [col for col, freq in outlier_counter.items() if freq == max_frequency]

                        # print("Individual Outliers:", outliers)
                        # print("Outlier Counts:", outlier_counter)
                        # print("Most Frequent Outliers:", most_frequent_outliers)

                    except Exception as e:
                        print(f"An error occurred: {e}")

                    # Return the most frequent outlier(s) as a comma-separated string
                    return ", ".join(most_frequent_outliers)

                column_output_name = op.get("column_output_name") or f"{column_name} Outlier"
                # Apply the most frequent outlier detection to each row
                combined_data[column_output_name] = combined_data.apply(most_frequent_outlier_for_row, axis=1,
                                                                        outlier_column=column_name_pair)

                combined_data = combined_data.drop(column_name_pair, axis=1)
            # for response that has variation, for each scorer (e.g., "Correctness Answer", "Helpfulness"), we calculate the difference between columns as a "trend"

        data_configs = self.config.ComparisonSchema.DataConfigs
        column_name_mapping = data_configs.get("column_map", {})

        scorer_names = column_name_mapping.get("scorers")
        for scorer in scorer_names:
            scorer_columns = self.comparison_group[scorer]
            trend_values = []
            trend_columns = []

            for i in range(1, len(scorer_columns)):
                col1 = f"{scorer}.{i}"
                col2 = f"{scorer}.{i + 1}"
                trend_column_name = f"{scorer} Trend {i}_to_{i + 1}"
                combined_data[trend_column_name] = pd.to_numeric(combined_data[col2], errors='coerce') - pd.to_numeric(
                    combined_data[col1], errors='coerce')

                # Keep track of trend columns to delete later
                trend_columns.append(trend_column_name)

            # Store the list of trend values in a unified column
            combined_data[f"{scorer} Trend"] = combined_data[trend_columns].values.tolist()

            combined_data = combined_data.drop(columns=trend_columns, axis=1)

            combined_data_copy = combined_data.copy()

            combined_data_copy[f"{scorer} Improvement"] = combined_data_copy[f"{scorer} Trend"].apply(is_increasing)
            combined_data_copy[f"{scorer} Regression"] = combined_data_copy[f"{scorer} Trend"].apply(is_decreasing)

            df_improvement = combined_data_copy[combined_data_copy[f"{scorer} Improvement"]].copy()
            df_regression = combined_data_copy[combined_data_copy[f"{scorer} Regression"]].copy()

            comparison_results.update({
                f"{scorer} Improvement": df_improvement.to_dict(),
                f"{scorer} Regression": df_regression.to_dict(),
            })

        # Summarize outliers for each responsen_name_pair)
        def summarize_outliers(row, column_name):
            if row[column_name]:
                outlier_columns = row['Response Outlier'].split(', ')
                trend_summary = []

                # Summarize trends for each scorer
                for scorer in scorer_names:
                    trends = []
                    scorer_columns = self.comparison_group[scorer]
                    trends = row[f"{scorer} Trend"]

                    # Determine overall trend pattern
                    if trends[-1] != 0 and all(t >= 0 for t in trends):
                        trend_description = f"{scorer}: trending up"
                        trend_summary.append(trend_description)
                    elif trends[-1] != 0 and all(t <= 0 for t in trends):
                        trend_description = f"{scorer}: trending down"
                        trend_summary.append(trend_description)

                if trend_summary:
                    return f"{', '.join(outlier_columns)}: " + "; ".join(trend_summary)
            return None

        if 'Response Outlier' in combined_data.columns:
            combined_data['Outlier Scores Shift'] = combined_data.apply(summarize_outliers, axis=1,
                                                                        column_name='Response Outlier')

        return combined_data, comparison_results

    def _calculate_similarity(self, series1: pd.Series, series2: pd.Series) -> Dict[str, float]:
        """
        Calculate TF-IDF and BERT similarities between two series.
        """
        tfidf_sim = self._calculate_tfidf_similarity(series1, series2)
        bert_sim = self._calculate_bert_similarity(series1, series2)

        return {
            "tfidf_similarity": tfidf_sim,
            "bert_similarity": bert_sim
        }

    @staticmethod
    def _calculate_tfidf_similarity(series1: pd.Series, series2: pd.Series) -> float:
        """
        Calculate TF-IDF based cosine similarity between two series.
        """
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(series1.astype(str) + ' ' + series2.astype(str))
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    @staticmethod
    def _calculate_bert_similarity(series1: pd.Series, series2: pd.Series) -> float:
        """
        Calculate BERT-based similarity (F1 score) between two series.
        """
        _, _, f1 = score(series1.astype(str).tolist(), series2.astype(str).tolist(),
                         lang="en", model_type="bert-base-uncased", batch_size=32)
        return f1.mean().item()

    def _generate_outliers(self, results: Dict[str, Any]) -> Dict[str, List[bool]]:
        """
        Generate outlier labels based on similarity results.
        """
        outliers = {}
        for col, similarities in results.items():
            tfidf_scores = [sim['tfidf_similarity'] for sim in similarities.values()]
            bert_scores = [sim['bert_similarity'] for sim in similarities.values()]

            tfidf_outliers = self._detect_outliers(tfidf_scores)
            bert_outliers = self._detect_outliers(bert_scores)

            outliers[col] = [t or b for t, b in zip(tfidf_outliers, bert_outliers)]

        return outliers

    def _detect_outliers(self, scores: List[float], threshold: float = 1.5) -> List[bool]:
        """
        Detect outliers using the Interquartile Range (IQR) method.
        """
        threshold = 0.7

        return []


# name: response, type: string, value: bert_f1

def add_similarity_score(df: pd.DataFrame, col1: str, col2: str,
                         score_function: Callable[[pd.DataFrame, str, str, str], pd.DataFrame],
                         score_col: str) -> pd.DataFrame:
    """
    Adds a similarity score between two columns to the DataFrame using the provided scoring function.

    :param df: The DataFrame containing the columns to compare.
    :param col1: The name of the first column.
    :param col2: The name of the second column.
    :param score_function: A function to compute similarity scores (e.g., apply_bert_f1).
    :param score_col: The name of the new column where the score will be stored.
    :return: The updated DataFrame with the new similarity score column.
    """
    # Call the provided similarity score function
    df = score_function(df, col1, col2, score_col)

    return df


def apply_bert_f1(df: pd.DataFrame, col1: str, col2: str, score_col: str) -> pd.DataFrame:
    """
    Adds a new column with BERT F1 scores to the DataFrame.

    :param df: DataFrame containing the two columns to compare.
    :param col1: Name of the first column (ground truth).
    :param col2: Name of the second column (predicted).
    :param score_col: Name of the column to store the BERT F1 score.
    :return: DataFrame with the BERT F1 score column added.
    """
    # Extract the ground truth and predicted columns from the DataFrame
    gt_list = [str(c) for c in df[col1].tolist()]
    pt_list = [str(c) for c in df[col2].tolist()]

    # Calculate BERT F1 scores
    _, _, f1 = score(gt_list, pt_list, lang="en", model_type="bert-base-uncased", batch_size=64)

    # Convert the torch.Tensor to a list and add it as a new column
    df[score_col] = f1.tolist()

    return df


def apply_tfidf_similarity(df: pd.DataFrame, col1: str, col2: str, score_col: str) -> pd.DataFrame:
    """
    Adds a new column with TF-IDF cosine similarity scores to the DataFrame.

    :param df: DataFrame containing the two columns to compare.
    :param col1: Name of the first column.
    :param col2: Name of the second column.
    :param score_col: Name of the column to store the TF-IDF similarity score.
    :return: DataFrame with the TF-IDF similarity score column added.
    """
    # Extract the two columns as lists of strings
    col1_list = [str(c) for c in df[col1].tolist()]
    col2_list = [str(c) for c in df[col2].tolist()]

    # Combine both lists for TF-IDF vectorization
    combined_texts = col1_list + col2_list

    # Compute TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_texts)

    # Split the TF-IDF matrix back into two parts
    tfidf_col1 = tfidf_matrix[:len(col1_list)]
    tfidf_col2 = tfidf_matrix[len(col1_list):]

    # Compute cosine similarity between the two sets of vectors
    similarities = cosine_similarity(tfidf_col1, tfidf_col2)

    # Take the diagonal similarity values (same index comparison)
    df[score_col] = similarities.diagonal()

    return df


def calculate_significance(statistics_current, statistics_reference, alpha=0.01):
    """
    Perform a paired t-test to see if there's a statistically significant difference between
    statistics_current and statistics_reference.

    @param statistics_current: A list or series of numbers summarizing the responses from df_current
    @param statistics_reference: A list or series of numbers summarizing the responses from df_reference
    """
    # Two groups of ratings from the same subjects
    try:
        group1_ratings = np.array(statistics_current)
        group2_ratings = np.array(statistics_reference)

        # Perform paired t-test (within-subject t-test)
        t_statistic, p_value = stats.ttest_rel(group1_ratings, group2_ratings, nan_policy="omit")

        # Print the results
        print("T-statistic:", t_statistic)
        print("P-value:", p_value)

        # Check if the difference is statistically significant
        # if alpha and p_value < alpha:
        return t_statistic, p_value

    except Exception as e:
        print(f"Error performing paired t-test: {e}")


def calculate_trend_direction(
        df: pd.DataFrame,
        scorer: List[str],
        source_column: str
) -> pd.DataFrame:
    """
    Determines the trend direction for each scorer based on mean values of each position in the trend lists per source.

    Parameters:
    df (pd.DataFrame): DataFrame containing the source column and trend columns.
    scorer (List[str]): List of column names without the "Trend" suffix.
    source_column (str): Name of the source column in the DataFrame.

    Returns:
    pd.DataFrame: DataFrame where each row is a source and columns are trend directions for each scorer.
    """
    trend_results: Dict[str, Dict[str, str]] = {}

    # Iterate over each scorer to calculate trend direction
    for score in scorer:
        trend_column = f"{score} Trend"

        # Group by source to assess trend direction within each group
        for source, group in df.groupby(source_column):
            # Collect each trend position separately
            trend_positions = list(zip(*group[trend_column].tolist()))

            # Calculate mean for each position
            means = [sum(position) / len(position) for position in trend_positions]

            # Determine trend direction based on the means of each position
            if all(mean >= 0 for mean in means):
                trend_direction = "trending up"
            elif all(mean <= 0 for mean in means):
                trend_direction = "trending down"
            else:
                trend_direction = ""

            # Update the trend results dictionary
            if source not in trend_results:
                trend_results[source] = {}
            trend_results[source][f"{score} Trend"] = trend_direction

    # Convert trend_results to a DataFrame for better readability and usage
    trend_df = pd.DataFrame.from_dict(trend_results, orient='index').reset_index()
    trend_df.rename(columns={'index': source_column}, inplace=True)

    return trend_df


@log_errors("Save evaluation results")
def _save_evaluation_history_to_database(request: RunMakerRequest, datastore_content: dict,
                                         evaluation_result: Any, status: EvaluationRunStatus.STARTED) -> None:
    response = None
    try:
        evaluation_history_id = DatabaseHandler({'env': 'dev'}).add_to_evaluation_history(
            run_request=request,
            dataset_metadata=datastore_content,
            evaluation_metadata=request.EvaluationConfig.dict() if request.EvaluationConfig else {},
            flowconfig_metadata=request.PlatformConfig.dict() if request.PlatformConfig else {},
            output_url=json.dumps(evaluation_result),
            status=str(status.value)
        )
        print(f"Saved evaluation history in database: {evaluation_history_id}")
        logger.info(f"Saved evaluation history in database: {evaluation_history_id}")

        # verify
        response = DatabaseHandler({'env': 'dev'}).get_evaluation_history(
            history_id=int(evaluation_history_id),
        )
        return response
    except Exception as ex:
        print(f"Exception in save evaluation history to database, {ex}")
        raise Exception(ex)
    return response


@log_errors("Divide columns by response")
def _divide_columns_by_response(column_names):
    # Check if "Response" and "Duration" exist in the list
    if "Response" not in column_names:
        return [], []  # Return two empty lists if either column is missing

    # Find the index of "Response" and "Duration"
    response_index = column_names.index("Response")
    # duration_index = column_names.index("Duration")

    # Get columns before "Response" and between "Response" and "Duration"
    before_response = column_names[:response_index]
    after_response = column_names[response_index:]

    return before_response, after_response


def _reorder_columns(df):
    # Extract base columns and suffixed columns
    base_columns = [col for col in df.columns if not re.search(r"\.\d+$", col)]
    suffixed_columns = [col for col in df.columns if re.search(r"\.\d+$", col)]

    # Group suffixed columns by their base name
    from collections import defaultdict

    grouped_columns = defaultdict(list)
    for col in suffixed_columns:
        base_name = col.rsplit(".", 1)[0]
        grouped_columns[base_name].append(col)

    # Sort each group by the suffix and flatten the list
    sorted_suffixed_columns = []
    for base_name in sorted(grouped_columns):  # Sort by base name
        sorted_suffixed_columns.extend(sorted(grouped_columns[base_name], key=lambda x: int(x.rsplit(".", 1)[1])))

    # Combine base columns and sorted suffixed columns
    ordered_columns = base_columns + sorted_suffixed_columns

    # Reorder the DataFrame columns
    return df[ordered_columns]


def _construct_created_timestamp_str():
    pacific_tz = pytz.timezone('US/Pacific')
    utc_time = datetime.utcnow().replace(tzinfo=pytz.utc)  # Ensure the datetime is timezone-aware

    # Convert the UTC datetime to Pacific Time
    created_timestamp = utc_time.astimezone(pacific_tz)
    created_timestamp_str = created_timestamp.strftime('%Y-%m-%d_%H:%M:%S_%Z')

    return created_timestamp_str


def _check_and_trim_dfs(dfs_dict):
    """
    Check if the DataFrames in the dictionary have different lengths, print a message if so,
    and trim all DataFrames to the minimum length, without changing the keys.

    Parameters:
        dfs_dict (dict): Dictionary of pandas DataFrames.

    Returns:
        dict: Dictionary with trimmed DataFrames.
    """
    # Get the lengths of all DataFrames
    lengths = {key: len(df) for key, df in dfs_dict.items()}
    min_length = None

    # Check if all lengths are the same
    if len(set(lengths.values())) > 1:  # True
        print("The DataFrames have different lengths:", lengths)

        # Find the minimum length
        if not min_length:
            min_length = min(lengths.values())

        # Trim all DataFrames to the minimum length
        trimmed_dfs_dict = {key: df.iloc[:min_length].reset_index(drop=True) for key, df in dfs_dict.items()}
        print(f"Trimmed all DataFrames to the minimum length: {min_length}")
        return trimmed_dfs_dict

    return dfs_dict
