import copy
import logging
import random
import re
import statistics
import string
from datetime import datetime
from typing import Tuple, List, Any


import requests
import huggingface_hub as hh
import uuid
import json
import os

import shortuuid
from fastapi import status as http_status
from tqdm import tqdm

from configs.settings import get_settings
from data_models.api.run_maker import NVBotEvaluationConfig, RunMakerRequest
from nvbot_utilities.utils.utilities import get_class_module, create_func_instance
from service_library.constants import CUSTOM_EVAL_INPUT_FILENAME, CUSTOM_EVAL_OUTPUT_FILENAME, \
    EVALUATION_CONFIG_DEFAULT, EVAL_INFERENCE_DEFAULT, \
    JUDGE_EVAL_INFERENCE_PARAMS_DEFAULT, DATASET_FILENAME
from service_library.utils.azure_helper import get_oauth_client_token, get_prd_oauth_client_token
from service_library.utils.data_helper import jsonl_writer, json_writer
from service_library.utils.logging import log_errors
from service_library.utils.run_helpers import generate_excel_output, create_folder_if_not_exists, convert_to_snake_case

DS_URL = "https://datastore.dev.llm.ngc.nvidia.com"
EVAL_URL = "https://evaluation-ms-staging-nemo-evaluator.dev.llm.ngc.nvidia.com"

HEADERS = {
    # "Content-Type": "application/json",
    "Accept": "application/json"
}
TOKEN = "mock_token"

logger = logging.getLogger('Nemo Service Helper')


## helper functions


def get_models():
    get_datasets_endpoint = "{}/v1/models".format(DS_URL)
    print(get_datasets_endpoint)
    # params = {
    #     "page_size": 100,
    #     "page": 1
    #     }

    response = requests.get(get_datasets_endpoint, headers=HEADERS)
    print(response)
    assert response.status_code == http_status.HTTP_200_OK, response.text

    result = response.json()
    # print (f"{dataset_id} contains files: ", [file['path'] for file in result['files']])
    return [(model["id"], model["name"]) for model in result["models"]]


def get_dataset_contents(url: str, dataset_id: str, file_path_only: bool = False):
    get_datasets_endpoint = "{}/v1/datasets/{}".format(url, dataset_id)

    response = requests.get(get_datasets_endpoint, headers=HEADERS)
    assert response.status_code == http_status.HTTP_200_OK, response.text
    result = response.json()

    if file_path_only:
        files = [file['path'] for file in result['files'] if not file['path'].startswith(".")]
        print(f"files: {files}")
        return files
    return result


def get_evaluation_parameter_schema(url, type: str = "llm_as_a_judge", sub_type="custom_eval"):
    endpoint = f"{url}/v1/evaluation_configs"
    params = {
        "type": type,
        "subtype": sub_type
    }
    response = requests.get(endpoint, params=params)
    assert response.status_code == http_status.HTTP_200_OK, response.text
    evaluation_configs = response.json()['evaluations']
    return evaluation_configs


# repo_name = "<Value of the 'tag' parameter in the eval config>" for llm as a judge
@log_errors("Download results from datastore")
def download_results_to_local_directory(url, repo_id, local_directory="eval-results"):
    # Specify the local path to download the results to.
    # if not repo_id.startswith("nvidia/"):
    #     repo_id = f"nvidia/{repo_id}"
    # print("result download to local dir: ", local_directory)
    path = os.path.join(local_directory, repo_id)
    create_folder_if_not_exists(path)
    download_directory = path or local_directory

    api = hh.HfApi(endpoint=url, token="token_mock")
    # Download the results into the current directory.
    download_response = api.snapshot_download(
        repo_id=f"nvidia/{repo_id}",
        repo_type='dataset',
        cache_dir=None,
        local_dir=download_directory,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to {download_response}")
    logger.info(f"Downloaded {repo_id} to {download_response}")
    return download_directory


def json_to_jsonl(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    print("data", data)
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


@log_errors("Create datastore dataset entity")
def create_dataset(url, repo_name, description=""):
    ds_id = None
    endpoint = f"{url}/v1/datasets"
    params = {
        "name": repo_name,
        "description": description or "dataset",
    }
    response = requests.post(endpoint, json=params).json()

    if 'id' in response:
        ds_id = response['id']
    else:
        print("failed create dataset", response)
        raise Exception(f"failed create dataset, response {response}")
    return ds_id


@log_errors("Create datastore dataset and upload")
def create_dataset_and_upload_folder(url, repo_name, local_dir_path, path_in_repo="."):
    """
    repo_name needs to be unique
    path in repo will be created if not exist
    """
    # check repo_name
    if len(repo_name) >= 96:
        repo_name = repo_name[:80] + '-' + generate_random_string(5)
    dataset_id = create_dataset(url, repo_name)
    if dataset_id is None:
        print("dataset exist for repo_name")
    logger.info(f"created dataset id: {dataset_id}")
    print(f"created dataset id: {dataset_id}")

    # upload dir
    repo_full_name = f"nvidia/{repo_name}"
    # print("repo_full_name:", repo_full_name)
    # print ("url")
    # path_in_repo = f".{path_in_repo}" if path_in_repo.startswith("/") else f"./{path_in_repo}"
    # print("path_in_repo:", path_in_repo)
    # print("local_dir_path:", local_dir_path)

    hf_api = hh.HfApi(endpoint=url, token="token12345")

    upload_response = hf_api.upload_folder(
        repo_id=repo_full_name,
        folder_path=local_dir_path,
        path_in_repo=path_in_repo,
        repo_type="dataset"
    )
    print("upload complete", upload_response)
    return dataset_id, repo_full_name


# def run_eval(repo_name, llm_params=None):
#     if llm_params is None:
#         llm_params = build_llm_params(repo_name)
#     endpoint = f"{EVAL_URL}/v1/evaluations"
#     response = requests.post(endpoint, json=llm_params).json()
#     evaluation_id = response["evaluation_id"]
#     print("Evaluation response\n", response)
#     logging.info(f"Evaluation id: {evaluation_id}")
#     return response


def launch_evaluation(url, params):
    endpoint = f"{url}/v1/evaluations"
    print("params:", params)
    response = requests.post(endpoint, json=params)
    print("Nemo Evaluation response:\n", response.text)
    assert response.status_code == http_status.HTTP_201_CREATED, response.text
    result = response.json()
    print("Nemo Evaluation result:\n", result)
    print(f"\nNemo Evaluation id: {result['evaluation_id']}")
    return result


def check_eval_result(url, eval_id):
    print("eval_id: ", eval_id)
    endpoint = f"{url}/v1/evaluations/{eval_id}"
    response = requests.get(endpoint, headers=HEADERS).json()

    return response


def parse_custom_eval_aggregated_metrics(data):
    """
    return results in dictionary format
    e.g
        {
          "name": "rouge_1_score",
          "value": 0.2836645543575287,
          "metadata": {
            "llm_name": "mistralai/mixtral-8x22b-instruct-v0.1",
            "scorer": "rouge"
          }
        }
    """
    parsed_data = {}
    for item in data:
        value = item['value']
        if isinstance(value, (int, float)):
            value = round(float(item['value']), 2)
        parsed_data[item['name']] = value
    return parsed_data


def parse_llm_as_a_judge_eval_aggregated_metrics(data):
    """
    return results in dictionary format
    e.g {"bleu": 11}
    """
    parsed_data = {}

    for task in data.get("task_results", []):
        task_name = task["name"].strip().replace(" ", "_")
        for metric in task.get("metrics", []):
            metric_value = metric["value"]
            # Convert the metric value to float, defaulting to 0.0 if conversion fails
            try:
                metric_value = round(float(metric_value), 2)
            except ValueError:
                metric_value = None
            # Add the parsed result to the dictionary
            parsed_data[task_name] = metric_value

    return parsed_data


#
# def build_llm_params(repo_name, eval_type="llm_as_a_judge"):
#     return {
#         "model": {
#             "llm_type": "openai-schema-base",
#             "llm_name": "gpt-43b-002",
#             # "inference_url": "http://gpt-8b-nemollm-inference.nemo-evaluation.svc.cluster.local:8006/v1",
#             "inference_url": "https://devbot-api.nvidia.com/evaluation",
#             "is_chat_model": False
#         },
#         "evaluations": [
#             {
#                 "eval_type": "llm_as_a_judge",
#                 "eval_subtype": "mtbench",
#                 "bench_name": "not_mt_bench",
#                 "mode": "single",
#                 "input_dir": f"nds:{repo_name}",
#
#                 "inference_params": {
#                     "top_p": 0.9,
#                     "top_k": 40,
#                     "temperature": 0.75,
#                     "stop": [],
#                     "tokens_to_generate": 1024
#                 },
#
#                 "judge_model": {
#                     "llm_type": "nvidia-nemo-nim",
#                     "llm_name": "nv-gpt-8b-base",
#                     "inference_url": "http://gpt-8b-nemollm-inference.nemo-evaluation.svc.cluster.local:8006/v1",
#                     "is_chat_model": False,
#                 },
#
#                 "judge_inference_params": {
#                     "top_p": 0.9,
#                     "top_k": 1,
#                     "temperature": 0.0,
#                     "stop": [],
#                     "tokens_to_generate": 1024,
#                 }
#             }
#         ],
#         "tag": repo_name,
#     }


def generate_random_string(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


DEFAULT_COLUMN_MAP = {
    "question": "Question",
    "reference": "Reference",
    "answer": "Answer",
    "category": "Category",
    # Note: judge_prompt is used only in eval_type as "llm_as_a_judge"
}


def prepare_custom_nemoeval_dataset(local_folder, df, evaluator_config, run_file=None, limit=None,
                                    ):
    # TODO, "column_map" to type
    column_params_map = evaluator_config.get('column_map') or DEFAULT_COLUMN_MAP
    question_header = column_params_map['question'] or DEFAULT_COLUMN_MAP['question']
    reference_header = column_params_map['reference'] or DEFAULT_COLUMN_MAP['reference']
    answer_header = column_params_map['answer'] or DEFAULT_COLUMN_MAP['answer']
    category_header = column_params_map.get('category') or DEFAULT_COLUMN_MAP['category']

    model_config = evaluator_config.get("model")
    llm_name = model_config.get("llm_name", "mixtral-8x7b")

    # if df is None:
    #     df = pd.read_csv(run_file, encoding='utf-8')
    df = df.fillna("")
    sample = df.copy()
    if limit:
        sample = df.iloc[:int(limit)]
        # reference = df.iloc[divider:]

    # generate temp folder, prepare this for upload
    current_path = os.getcwd()

    input_template = {
        "prompt": "",
        "category": "Closed QA",
        "ideal_response": "",
        "source": ""}

    output_template = {
        "input": input_template.copy(),
        "response": "",
        "llm_name": llm_name,
    }

    # build question file
    inputs = []
    outputs = []
    for q_index, row in tqdm(sample.iterrows(), total=df.shape[0]):
        # Create a new dictionary for each turn
        try:
            if question_header not in row:
                error_message = f"Question header {question_header} not exist, row index: {q_index}"
                logging.error(error_message)
                print(error_message)
                continue
            elif not row[question_header]:
                error_message = f"Question header {question_header} is not acceptable, value: {row[question_header]}"
                logging.error(error_message)
                print(error_message)
                continue

            if answer_header not in row:
                error_message = f"Answer header {answer_header} not exist, row index: {q_index}"
                logging.error(error_message)
                print(error_message)
                continue
            elif not row[answer_header]:
                error_message = f"Answer header {answer_header} is not acceptable, value: {row[answer_header]}"
                logging.error(error_message)
                print(error_message)
                continue

            if reference_header not in row:
                error_message = f"Reference header {reference_header} not exist, row index: {q_index}"
                logging.error(error_message)
                print(error_message)
                continue
            elif not row[reference_header]:
                error_message = f"Reference header {reference_header} is not acceptable, value: {row[reference_header]}"
                logging.error(error_message)
                print(error_message)
                continue

            # build input
            entry = copy.deepcopy(input_template)
            entry['prompt'] = row[question_header] or ""

            # if category_header in row:
            #     entry['category'] = row[category_header]

            if reference_header in row:
                entry['ideal_response'] = str(row[reference_header]) or ""
            inputs.append(entry)

            # build output
            output_entry = copy.deepcopy(output_template)
            output_entry['input'] = entry.copy()
            output_entry['response'] = row[answer_header] or ""

            outputs.append(output_entry)


        except Exception as ex:
            logging.error(f"Row: {row}, exception: {ex}")

    print("Sample inputs: ", inputs[0])
    print("Sample outputs: ", outputs[0])

    # jsonl_writer(inputs, os.path.join(current_path, local_folder, inputs_filename))
    # jsonl_writer(inputs, os.path.join(current_path, local_folder, outputs_filename))

    json_writer(inputs, os.path.join(current_path, local_folder, CUSTOM_EVAL_INPUT_FILENAME))
    json_writer(outputs, os.path.join(current_path, local_folder, CUSTOM_EVAL_OUTPUT_FILENAME))


def prepare_llm_as_a_judge_nemoeval_dataset(local_folder, df, evaluator_config, run_file=None, limit=None,
                                            ):
    column_params_map = evaluator_config.get('column_map') or DEFAULT_COLUMN_MAP
    question_header = column_params_map.get('question') or DEFAULT_COLUMN_MAP['question']
    reference_header = column_params_map.get('reference') or DEFAULT_COLUMN_MAP['reference']
    answer_header = column_params_map.get('answer') or DEFAULT_COLUMN_MAP['answer']
    prompt_header = column_params_map.get('prompt')

    model_config = evaluator_config.get("model")
    llm_name = model_config.get("llm_name", "mixtral-8x7b")

    # if df is None:
    #     df = pd.read_csv(run_file, encoding='utf-8')
    df = df.fillna("")
    sample = df.copy()
    print("limit", limit)
    if limit:
        sample = df.iloc[:int(limit)]
        # reference = df.iloc[divider:]

    # generate temp folder, prepare this for upload
    current_path = os.getcwd()
    create_folder_if_not_exists(os.path.join(current_path, local_folder))
    df.to_excel(os.path.join(current_path, local_folder, DATASET_FILENAME), index=False)

    question_template = {"question_id": 1, "category": "general", "turns": []}

    # build question file
    questions = []
    for q_index, row in tqdm(sample.iterrows(), total=df.shape[0]):
        # Create a new dictionary for each turn
        try:
            if question_header not in row:
                error_message = f"Question header {question_header} not exist, row index: {q_index}"
                logging.error(error_message)
                print(error_message)

            entry = copy.deepcopy(question_template)
            entry['turns'].append(str(row.get(question_header, "")) or "")
            entry['question_id'] = q_index
            questions.append(entry)
        except Exception as ex:
            logging.error(f"Row: {row}, exception: {ex}")

    print("Sample question: ", questions[0])

    question_filename = "question.jsonl"
    jsonl_writer(questions, os.path.join(current_path, local_folder, question_filename))

    # build ref file
    reference_folder = "reference_answer"
    reference_template = {"question_id": 1, "answer_id": shortuuid.uuid(),
                          "model_id": llm_name, "choices": [{"index": 0, "turns": []}], "tstamp": 1686286924.844282}
    create_folder_if_not_exists(os.path.join(current_path, local_folder, reference_folder))

    references = []
    for re_index, row in tqdm(sample.iterrows(), total=df.shape[0]):

        try:
            entry = copy.deepcopy(reference_template)
            turns = entry['choices'][0]['turns']

            if reference_header not in row:
                error_message = f"Reference header {reference_header} not exist, row index: {re_index}"
                logging.error(error_message)
                print(error_message)

            turns.append(str(row.get(reference_header, "")) or "")
            choices0 = entry['choices'][0]
            choices0['turns'] = turns
            entry['choices'][0] = choices0
            entry['question_id'] = re_index
            entry['tstamp'] = datetime.timestamp(datetime.now())

            references.append(entry)

        except Exception as ex:
            logging.error(f"Row: {row}, exception: {ex}")

    print("Sample reference: ", references[0])

    reference_filename = "reference.jsonl"
    jsonl_writer(references, os.path.join(current_path, local_folder, reference_folder, reference_filename))

    # build answer
    answer_folder = "model_answer"
    answer_template = {"question_id": 1, "answer_id": shortuuid.uuid(),
                       "model_id": "", "choices": [{"index": 0, "turns": []}], "tstamp": 1686286924.844282}
    create_folder_if_not_exists(os.path.join(current_path, local_folder, answer_folder))

    answers = []
    for ans_index, row in tqdm(sample.iterrows(), total=df.shape[0]):

        try:
            entry = copy.deepcopy(answer_template)
            turns = entry['choices'][0]['turns']

            if answer_header not in row:
                error_message = f"Answer header {answer_header} not exist, row index: {ans_index}"
                logging.error(error_message)
                print(error_message)
                continue

            turns.append(row[answer_header] if answer_header in row else "Answer appropriately")
            choices0 = entry['choices'][0]
            choices0['turns'] = turns
            entry['choices'][0] = choices0
            entry['question_id'] = ans_index
            entry['tstamp'] = datetime.timestamp(datetime.now())

            answers.append(entry)

        except Exception as ex:
            logging.error(f"Row: {row}, exception: {ex}")

    print("Sample answer: ", answers[0])

    answer_filename = "nvbot-adapter.jsonl"
    jsonl_writer(answers, os.path.join(current_path, local_folder, answer_folder, answer_filename))

    # TODO: get judge config

    # build parameters folder

    # Create a folder called "judge_prompt_parameters"
    judge_prompt_parameters_folder = 'judge_prompt_parameters'
    create_folder_if_not_exists(os.path.join(current_path, local_folder, judge_prompt_parameters_folder))

    # Iterate over the extract_keys and extract corresponding data from the Excel file

    additional_params = column_params_map.get("additional_params", [])
    print(f"create subfolder under {judge_prompt_parameters_folder} folder: {additional_params} ")
    for key in additional_params:
        values = []
        snake_case_key = convert_to_snake_case(key)
        if key in df.columns:
            # Convert key to snake case

            for index, value in enumerate(df[key]):
                data = {
                    "question_id": index,
                    "value": str(value)
                }
                values.append(data)
        else:
            for index in range(df.shape[0]):
                data = {
                    "question_id": index,
                    "value": ""
                }
                values.append(data)
        jsonl_writer(values, os.path.join(current_path, local_folder, judge_prompt_parameters_folder,
                                              f"{snake_case_key}.jsonl"))

    # build judge file
    judge_template = {
        "name": "single-ref-v1",
        "type": "single",
        "prompt_template": "",
        "description": "for general LLM response evaluation",
        "category": "general",
        "output_format": "",
        "system_prompt": "You are a helpful assistant."
    }

    # outputformat = "prompt_library.metrics_eval_prompt.MetricsEvaluationPrompt.output_format_v2",
    prompt_config = evaluator_config.get("judge_config")
    if not prompt_config:
        return
    prompt_module = prompt_config.get("prompt_module") or EVALUATION_CONFIG_DEFAULT["prompt_module"]
    module_name, function_name = get_class_module(prompt_module)
    func = create_func_instance(module_name=module_name, class_name=function_name)

    output_format_attribute = prompt_config.get("output_format") or EVALUATION_CONFIG_DEFAULT["output_format"]
    prompt_attribute = prompt_config.get("template") or EVALUATION_CONFIG_DEFAULT["template"]

    output_format = getattr(func, output_format_attribute)
    prompt = getattr(func, prompt_attribute)

    judge_prompts = []
    judge_copy = copy.deepcopy(judge_template)
    judge_copy['prompt_template'] = prompt
    judge_copy['output_format'] = output_format
    judge_prompts.append(judge_copy)

    # print("Judge prompts: ", judge_prompts[0])
    judge_prompt_filename = "judge_prompts.jsonl"
    jsonl_writer(judge_prompts, os.path.join(current_path, local_folder, judge_prompt_filename))


def get_extract_run_maker_request(request: RunMakerRequest,
                                  orig_dict: dict = {}):  # from request, extra keys in snake_case and append non-null value
    for key, value in extract_nemo_eval_metadata(request).items():
        if value is not None:
            orig_dict.update({convert_to_snake_case(key): value})
    return orig_dict


@log_errors("Custom Evaluation Payload Generation")
def prepare_custom_nemoeval_payload(url, dataset_config: dict, evaluator_payload: dict, model_config: dict,
                                    request: RunMakerRequest):
    print(f"prepare_custom_nemoeval_payload with dataset_config {dataset_config}")
    files = dataset_config.get("Files", [])
    # Filter for files ending with 'input.json' or 'input.jsonl'
    input_files = [path for path in files if path.endswith("input.jsonl") or path.endswith(CUSTOM_EVAL_INPUT_FILENAME)]
    print("input_files", input_files)
    # Filter for files ending with 'output.json' or 'output.jsonl'
    output_files = [path for path in files if
                    path.endswith("output.jsonl") or path.endswith(CUSTOM_EVAL_OUTPUT_FILENAME)]

    assert len(input_files) == 1, f"Expected 1 input file in format such as `{CUSTOM_EVAL_OUTPUT_FILENAME}`"

    # parameter_schema = get_evaluation_parameter_schema(
    #     url,
    #     evaluator_config.get("eval_type"),
    #     evaluator_config.get("eval_subtype", "")
    # )[0]

    inference_configs = evaluator_payload.get("inference_configs", [])
    inference_config = {}
    run_inference = False
    if len(inference_configs) >= 0:
        inference_config = inference_configs[0]
        run_inference = inference_config.get("run_inference", False)

    print("run_inference", run_inference)

    has_output = len(output_files) != 0
    use_nvcf_endpoint = True # url.endswith("7331")

    llm_name = model_config.get("llm_name", "mixtral-8x22b")
    inference_model_specification = nvcf_model_mapper(llm_name) if use_nvcf_endpoint else nemo_model_mapper(llm_name)
    path_in_repo = dataset_config.get("DatasetFolder")

    # TODO: move to constant
    nvidia_prefix = "nvidia/"
    output_folder_path = path_in_repo if not path_in_repo.startswith(nvidia_prefix) else path_in_repo[
                                                                                         len(nvidia_prefix):]
    input_path = os.path.join(output_folder_path, input_files[0])
    print("input_path:", input_path)

    if not run_inference:
        assert len(output_files) >= 1, "When not run inference, output files should be provided"
        print("output_files:", output_files[0])
        output_path = os.path.join(output_folder_path, output_files[0])
        evaluation_specification = {
            "eval_type": "automatic",
            "eval_subtype": "custom_eval",
            "input_file": f"nds:{input_path}",
            "inference_configs": [
                {
                    "model": {
                        "llm_name": inference_model_specification["llm_name"]
                    },
                    "run_inference": run_inference,
                    "output_file": f"nds:{output_path}",
                    "inference_params": {
                        **EVAL_INFERENCE_DEFAULT,
                        **inference_config.get("inference_params"),
                    }
                }
            ],
            "num_of_samples": -1,
            "scorers": evaluator_payload.get("scorers", ["accuracy", "bleu", "rouge", "em", "f1", "bert"]),
        }
    else:
        evaluation_specification = {
            "eval_type": "automatic",
            "eval_subtype": "custom_eval",
            "input_file": f"nds:{input_path}",
            "inference_configs": [
                {
                    "model": {
                        "llm_name": inference_model_specification["llm_name"],
                    },
                    "run_inference": run_inference,
                    "inference_params": {
                        **EVAL_INFERENCE_DEFAULT,
                        **inference_config.get("inference_params"),
                    }
                }
            ],
            "num_of_samples": -1,
            "scorers": ["accuracy", "bleu", "rouge", "em", "f1"],
        }
    # HACK: check succeed runs to copy over model
    model_specification = {
        "llm_name": "nvbot-adapter",
        "llm_type": "nvidia-nvcf-nemo-nim",
        "inference_url": "https://devbot-api.nvidia.com/evaluation/chat/completions",
        "is_chat_model": True
    } if use_nvcf_endpoint else nemo_model_mapper(llm_name)
    custom_eval_param = {
        "model": model_specification,
        "evaluations": [
            evaluation_specification
        ],
        "tag": f"nvbot-eval_{dataset_config.get('DatasetId')}"
    }
    return custom_eval_param


def prepare_llm_as_a_judge_nemoeval_payload(url, dataset_config: dict, evaluator_payload: dict, model_config: dict,
                                            request: RunMakerRequest):
    folder = dataset_config.get("DatasetFolder")
    nvidia_prefix = "nvidia/"
    input_folder_path = folder if not folder.startswith(nvidia_prefix) else folder[
                                                                            len(nvidia_prefix):]
    files = dataset_config.get("Files")
    judge_prompt_file = [path for path in files if path.endswith("judge_prompts.jsonl")]
    question_file = [path for path in files if path.endswith("question.jsonl")]

    if len(judge_prompt_file) != 1:
        raise Exception(f"Expected 1 judge prompt file in format such as `judge_prompts.jsonl`")
    if len(question_file) != 1:
        raise Exception(f"Expected 1 question file in format such as `question.jsonl`")

    # dont remove! TODO: Add check on config
    # parameter_schema = get_evaluation_parameter_schema(
    #     url,
    #     evaluator_config.get("eval_type"),
    #     evaluator_config.get("eval_subtype", "")
    # )

    use_nvcf_endpoint = True # url.endswith("7331")
    llm_name = model_config.get("llm_name", "mixtral-8x22b")

    # ! does not matter !
    model_specification = {
        "llm_name": "nvbot-adapter",
        "llm_type": "nvidia-nvcf-nemo-nim",
        "inference_url": "https://devbot-api.nvidia.com/evaluation/chat/completions",
        "is_chat_model": True
    } if use_nvcf_endpoint else nemo_model_mapper(llm_name)
    # ! does not matter completes !

    judge_model_specification = nvcf_model_mapper(llm_name)
    if not judge_model_specification:
        judge_model_specification = openai_model_mapper(llm_name)

    print ("llm_name:", llm_name)
    assert judge_model_specification is not None, f"Judge model not support: {llm_name}"

    judge_inference_params = evaluator_payload.get("judge_inference_params")
    inference_params = evaluator_payload.get("inference_params")

    extra_body = get_extract_run_maker_request(request, inference_params.get("extra_body", {}))

    inference_params["extra_body"] = extra_body

    return {
        "model": {
            **model_specification,
        },
        "evaluations": [
            {
                "eval_type": "llm_as_a_judge",
                "eval_subtype": "mtbench",
                "bench_name": "custom_bench",
                "mode": "single",
                "input_dir": f"nds:{input_folder_path}",
                "inference_params": {**EVAL_INFERENCE_DEFAULT, **inference_params},
                "judge_model": judge_model_specification,
                "judge_inference_params": {
                    **JUDGE_EVAL_INFERENCE_PARAMS_DEFAULT,
                    **judge_inference_params,
                }
            }
        ],
        "tag": f"nvbot-eval_{dataset_config.get('DatasetId')}"
    }


def extract_nemo_eval_metadata(run_request: RunMakerRequest):
    return {
        "System": run_request.System,  # required
        "Model": run_request.Model,  # required
        "ProjectId": run_request.ProjectId,
        "Project": run_request.Project,
        "UserId": run_request.UserId,
        "UserName": run_request.UserName,
        "Env": run_request.Env,
        "ConfigId": run_request.ConfigId
        # "Parameters": run_request.Parameters
    }

def convert_number(s: Any):
    if isinstance(s, (int, float)):  # If already a number, return as is
        return s
    s = s.strip()
    if s:
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                logger.error (f"Unable to parse {s} to numeric".format(s=s))
    return None


def extract_numeric_value_from_llm_judgment(text):
    # Clean up escaped quotes for JSON-like strings
    cleaned_text = re.sub(r'\\"', '"', text.strip())

    # Define regex patterns for matching quoted and unquoted key-value pairs
    pattern_quoted = re.compile(r'"([^"]+)":\s*([\d.]+)')
    pattern_unquoted = re.compile(r'^([\w\s]+?):\s*([\d.]+)', re.MULTILINE)

    # Try to find matches using the quoted pattern first
    quoted_matches = pattern_quoted.findall(cleaned_text)
    # print (1, matches)
    unquoted_matches = pattern_unquoted.findall(cleaned_text)
    # print (2, matches2)
    # If no matches found, use the pattern for unquoted key-value pairs

    # Convert matches to a dictionary
    result = {}
    for match in [quoted_matches, unquoted_matches]:
        for key, value in match:
            key = key.strip()
            result[key] = convert_number(value)

    return result


def extract_string_from_llm_judgment(json_string: str, parse_keys: List[str]):
    cleaned_string = re.sub(r'\\"', '"', json_string)
    pattern = re.compile(r'"([^"]+)":\s*"(.*?)"', re.DOTALL)
    matches = pattern.findall(cleaned_string)

    if not matches:
        # If no matches found, use pattern without quotes
        pattern = re.compile(r'([^\n:]+):\s*([^\n]*)', re.DOTALL)
        matches = pattern.findall(cleaned_string)

    result = {key: str(value) for key, value in matches if key in parse_keys}
    return result


def calculate_mean_and_std(scores):
    mean = statistics.mean(scores)
    std_dev = statistics.stdev(scores)
    return mean, std_dev

def openai_model_mapper(model_name):
    print ("openai_model_mapper: ", model_name)
    if "gpt-4" in model_name.lower():
        return {
            "llm_type": "openai-apicompatible",
            "llm_name": "gpt-4",
            "base_url": "https://test.api.nvidia.com/llm/v1/azure",
            "use_chat_endpoint": True,
            "api_key": get_oauth_client_token(),
            "use_cache": False,
            "is_hf_model": False
        }
    elif "o1" in model_name.lower():
        return {
            "llm_type": "openai-apicompatible",
            "llm_name": "o1-preview",
            "base_url": "https://prod.api.nvidia.com/llm/v1/azure",
            "use_chat_endpoint": True,
            "api_key": get_prd_oauth_client_token(),
            "use_cache": False,
            "is_hf_model": False
        }
    else:
        return None

def nvcf_model_mapper(model_name):
    if "mixtral-8x22b" in model_name.lower():
        return {
            "llm_type": "nvidia-nvcf-nemo-nim",
            "llm_name": "mistralai/mixtral-8x22b-instruct-v0.1",
            "container": "gitlab-master.nvidia.com:5005/swdl-nemollm-mlops/eval-tool:dkakwani-custom-dataset@sha256:9b1a74dbba2d5f60a4f026435be7917b37f3d3e98e9f1cc59c6748a09e397e27",
            "inference_url": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/6bebc771-bd0a-4ecc-8599-4987f2b19829",
            "is_chat_model": True,
            # "api_key": get_settings().PRIVATE_NVCF_API_KEY,
        }
    elif "llama-3.1-70b" in model_name.lower():
        return {
            "llm_type": "nvidia-nvcf-nemo-nim",
            "llm_name": "meta/llama-3.1-70b-instruct",
            "container": "gitlab-master.nvidia.com:5005/swdl-nemollm-mlops/eval-tool:dkakwani-custom-dataset@sha256:9b1a74dbba2d5f60a4f026435be7917b37f3d3e98e9f1cc59c6748a09e397e27",
            "inference_url": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/27b7db52-371d-4121-80e9-1ea0165b1a63",
            "is_chat_model": True,
            # "api_key": get_settings().PRIVATE_NVCF_API_KEY,
        }
    elif "llama3-70b-instruct" in model_name.lower():
        return {
            "llm_type": "nvidia-nvcf-nemo-nim",
            "llm_name": "public/meta/llama3-70b-instruct",
            "container": "gitlab-master.nvidia.com:5005/swdl-nemollm-mlops/eval-tool:dkakwani-custom-dataset@sha256:9b1a74dbba2d5f60a4f026435be7917b37f3d3e98e9f1cc59c6748a09e397e27",
            "inference_url": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/5ce17bfc-8328-40e6-9f53-3c896d403a77",
            "is_chat_model": True,
            # "api_key": get_settings().PRIVATE_NVCF_API_KEY,
        }
    elif "llama-3.2-3b" in model_name.lower():
        return {
            "llm_type": "nvidia-nvcf-nemo-nim",
            "llm_name": "public/meta/llama-3.2-3b-instruct",
            "container": "gitlab-master.nvidia.com:5005/swdl-nemollm-mlops/eval-tool:dkakwani-custom-dataset@sha256:9b1a74dbba2d5f60a4f026435be7917b37f3d3e98e9f1cc59c6748a09e397e27",
            "inference_url": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/4f1c926f-42aa-4b52-9ac9-c2a2098e432f",
            "is_chat_model": True
        }
    elif "llama-3.1-nemotron-70b" in model_name.lower():
        return {
            "llm_type": "nvidia-nvcf-nemo-nim",
            "llm_name": "public/nvidia/llama-3.1-nemotron-70b-instruct",
            "container": "gitlab-master.nvidia.com:5005/swdl-nemollm-mlops/eval-tool:dkakwani-custom-dataset@sha256:9b1a74dbba2d5f60a4f026435be7917b37f3d3e98e9f1cc59c6748a09e397e27",
            "inference_url": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/9b96341b-9791-4db9-a00d-4e43aa192a39",
            "is_chat_model": True
        }
    elif model_name.lower() == "adaptor":
        return {
            "llm_name": "nvbot-adapter",
            "inference_url": "https://devbot-api.nvidia.com/evaluation/chat/completions",
            "llm_type": "nvidia-nvcf-nemo-nim",
            "is_chat_model": True
        }
    else:
        return None


def nemo_model_mapper(model_name):
    # via https://evaluation.dev.llm.ngc.nvidia.com/v1/evaluations
    if model_name.lower() == "gpt-8b-base":
        return {
            "llm_name": "gpt-8b-base",
            "inference_url": "http://gpt-8b-base.nim-gpt-8b.svc.cluster.local:8006/v1",
            "llm_type": "nvidia-nemo-nim",
            # "is_chat_model": False
        }
    elif model_name.lower() == "gpt-43b-002":
        return {
            "llm_name": "gpt-43b-002",
            "inference_url": "http://gpt-43b-base.nim-gpt-43b.svc.cluster.local:8006/v1",
            "llm_type": "nvidia-nemo-nim",
            # "is_chat_model": False
        }
    elif model_name.lower() == "gpt-43b-905":
        return {
            "llm_name": "gpt-43b-905",
            "inference_url": "http://gpt-43b-905-nemollm-inference.nemollm-api.svc.cluster.local:8006/v1",
            "llm_type": "nvidia-nemo-nim",
        }
    ## stg
    elif model_name.lower() == "llama-2-70b-steerlm-chat":
        return {
            "llm_name": "llama-2-70b-steerlm-chat",
            "inference_url": "http://llama-2-70b-steerlm-chat.nemollm-api.svc.cluster.local:8006/v1",
            "llm_type": "nvidia-nemo-nim",
        }
    else:
        return {
            "llm_name": "gpt-8b",
            "inference_url": "https://gpt8b-ds.dev.llm.ngc.nvidia.com/v1",
            "llm_type": "nvidia-nemo-nim",
            # "is_chat_model": True,
        }
