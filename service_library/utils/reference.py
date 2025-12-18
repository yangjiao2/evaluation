# import json
# import logging
# import uuid
#
# from configs.settings import get_settings
# from data_models.common import EvaluationRequest
# from nvbot_models.constants.bot_configuration import PlatformAgentSelection
# from nvbot_models.request_models.userdata_request import UserEvaluationRunData
# from nvbot_utilities.utils import api_handler
#
# from utilities.common import utilities as ut
#
# log = logging.getLogger('ResultEvaluation')
#
#
# class ResultEvaluation:
#     def __init__(self, project: str, user_info: UserInfo):
#         self.project = project
#         self.user_info = user_info
#         self.prompt = ''
#         self.dataset_handler = DatasetHandler(project=self.project)
#
#     def run_evaluation(self, request: EvaluationRequest) -> str:
#         # Placeholder for running regression test
#         logging.info(f'create history record in db, {request.Project}')
#         try:
#
#             # create evaluation run in DB
#             # self._create_evaluation_history_record(request)
#
#             pass
#         except Exception as ex:
#             err_msg = f'Failed run evaluation, exception: {ex}'
#             print(err_msg)
#             logging.error(err_msg)
#         return ''
#
#     def run_regression(self, params: dict) -> str:
#         # Placeholder for running regression test
#         self.request = self._prepare_request(params)
#
#         logging.info(f'create history record in db for project {self.project}')
#         try:
#
#             # create evaluation run in DB
#             # self._create_evaluation_history_record(request)
#
#             pass
#         except Exception as ex:
#             err_msg = f'Failed run evaluation, exception: {ex}'
#             print(err_msg)
#             logging.error(err_msg)
#         return ''
#
#     def _default_dataset_address(self) -> str:
#         # Define default dataset address if none provided
#         # You can customize this method based on your requirements
#         return "/dataset/"
#
#     def _create_evaluation_history_record(self, request: EvaluationRequest) -> str:
#         status = False
#         try:
#             data = UserEvaluationRunData(
#                 NtAccount=request.UserInfo.UserId,
#                 Project=request.System,
#                 MetadataJson=request.Metadata,
#                 RunType=request.Runtype,
#                 Username=request.UserInfo.Username
#             )
#             url = f"{get_settings().NVBOT_SERVICES_URL}/evaluation_run"
#             log.info('create evaluation_history API URL: %s', url)
#             # headers = {
#             #     'Content-Type': 'application/json',
#             #     'Accept': 'application/json',
#             #     'device-initiated': 'false',
#             #     'Authorization': f'Bearer {user_token.get("token")}',
#             # }
#             headers = request.headers.get('Authorization')
#             response, status = api_handler.post_data(url=url, headers=headers, request_body=json.dumps(data))
#             return response, status
#
#         except Exception as ex:
#             log.error('Error submitting feedback %s', ex)
#
#         return status
#
#     def load_dataset_file(self, file_path: str, format='pd'):
#         # Convert DataFrame to Excel file in memory
#         try:
#             df = self.dataset_handler.download_file('dataset/' + file_path, format)
#             df.fillna('', inplace=True)  # fill NaN values with empty strings
#             headers = df.columns.tolist()
#             rows = df.values.tolist()
#             return {"headers": headers, "rows": rows}
#         except Exception as ex:
#             err_msg = f'Failed download: {file_path}, exception: {ex}'
#             print(err_msg)
#             logging.error(err_msg)
#         return ''
#
#     def list_file(self, directory: str, suffix='.xlsx'):
#         # Convert DataFrame to Excel file in memory
#         try:
#             return self.dataset_handler.list_files(directory, suffix)
#         except Exception as ex:
#             err_msg = f'Failed to load: {directory}, exception: {ex}'
#             print(err_msg)
#             logging.error(err_msg)
#         return ''
#
#     async def _arun(self):
#         project = self.get_user_input(self.project_prompt)
#         model = self.get_user_input(self.model_prompt)
#         dataset_address = self.get_user_input(self.dataset_prompt)
#         self.dataset_address = dataset_address
#         result = self.run_regression_test()
#         file_url = self.upload_to_s3(result)
#         print("Result uploaded to S3:", file_url)
#
#     def _prepare_request(self, metadata: dict):
#         def _get_source_from_metadata(metadata: dict):
#             source = metadata.get("source", "")
#             try:
#                 selection = PlatformAgentSelection(source)
#                 print("Selected platform:", selection)
#                 return selection
#             except ValueError:
#                 print(f"Invalid platform agent. Using default value. {PlatformAgentSelection.NVHELP}")
#             return PlatformAgentSelection.NVHELP
#
#         def _get_configs(metadata: dict):
#             ut.fetch_bot_config()
#
#         # query_id = uuid.uuid4()
#         bot_config = ut.fetch_bot_config(query_id, _get_source_from_metadata(metadata))  # PlatformAgentSelection
#         return EvaluationRequest(
#             Project=self.project,  # avc_mixtral
#             SourceSystem=metadata.get("source", ""),  # avc
#             Runtype=metadata.get("runType", "manual"),
#             Metadata=metadata,
#             UserInfo=self.user_info
#         )
#
#     # # Example usage:
#     # project_prompt = "Enter project name:"
#     # model_prompt = "Enter model name:"
#     # dataset_prompt = "Enter dataset address (leave empty for default):"
#     # llm_result_evaluator = LLMResultEvaluation(project_prompt, model_prompt, dataset_prompt)
#     # llm_result_evaluator.evaluate_results()
