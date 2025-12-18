##  filename constants ##
from nvbot_models.request_models.evaluation_request import EvaluationRunStatus

CUSTOM_EVAL_INPUT_FILENAME = "input.json"
CUSTOM_EVAL_OUTPUT_FILENAME = "output.json"
DATASET_FILENAME = "dataset.xlsx"
ANSWER_FILENAME = "answers.xlsx"

AGGREGATE_SCORES_FILENAME = "aggregate_scores.json"

## username

NT_ACCOUNT_ID = "yangj"
EVAL_ACCOUNT_ID = "nvbot-evaluation"
NT_ACCOUNT_NAME = "NVbot Evaluation"
NT_EMAIL_HANDLE = "nvbot-evaluation@exchange.nvidia.com"

## local folder

LOCAL_TMP_FOLDER_WITH_END_SLASH = "tmp/"
LOCAL_TMP_FOLDER = "tmp"
LOCAL_EVAL_RESULTS_TMP_FOLDER = "results_tmp"
LOCAL_EVAL_RESULTS_TMP_FOLDER_WITH_END_SLASH = "results_tmp/"

## evaluator type
LLM_AS_A_JUDGE_EVALUATOR_TYPE = "llm_as_a_judge"
CUSTOM_EVALUATOR_TYPE = "automatic"

LLM_AS_A_JUDGE_FILE_NAME = "answer_judgement.xlsx"
CUSTOM_FILE_NAME = "answers.xlsx"

## model
REFERENCE_MODEL = "gpt-4"

DEFAULT_JUDGE_PARAM = {
    "prompt_module": "eval_prompt_library.metrics_eval_prompt.MetricsEvaluationPrompt",
    "output_format": "output_format",
    "template": "eval_template",
    "scorers": [
        "Correctness Answer",
        "Helpfulness",
        "Empathy",
        "Conciseness"
    ],
    "parse_keys": [
        "Explanation"
    ]
}

## model config

EVAL_INFERENCE_DEFAULT = {
    "tokens_to_generate": 600,
    "temperature": 0,
    "top_k": 1,
    "top_p": 0.75,
    "stop": [],
}

JUDGE_EVAL_INFERENCE_PARAMS_DEFAULT = {
    "top_p": 0.1,
    "top_k": 40,
    "temperature": 0.0,
    "stop": [],
    "tokens_to_generate": 1024
}

EVALUATION_CONFIG_DEFAULT = {
    "prompt_module": "prompt_library.metrics_eval_prompt.MetricsEvaluationPrompt",
    "output_format": "judge-nvbot",
    "template": "eval_template"
}

EVAL_STATUS_MAPPING = {
    "failed": EvaluationRunStatus.FAILED,
    "succeeded": EvaluationRunStatus.COMPLETED,
    "running": EvaluationRunStatus.STARTED,
    "error": EvaluationRunStatus.ERROR
}
