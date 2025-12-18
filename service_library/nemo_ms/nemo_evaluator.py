import logging
import os
from typing import Optional

from service_library.nemo_ms.nemo_service_helper import launch_evaluation, check_eval_result
from service_library.utils.request_helpers import RequestType, make_request


log = logging.getLogger('Eval Router')
class NeMoEvaluator:
    def __init__(self, url, evaluator_config = None):
        self.url = url
        self.config = evaluator_config
        self.status = None
        self.req_result = None
        self.evaluation_content = {
            "id": None,
            "metadata": None
        }

    def launch(self):
        logging.info("Launching NeMo Evaluator")
        print (f"launch {self.url}, {self.config}")
        launch_response = launch_evaluation(self.url, self.config)
        eval_id = launch_response["evaluation_id"]
        self.evaluation_content = {
            "id": eval_id,
            "eval_type": launch_response["evaluations"][0].get("eval_type"),
            "eval_subtype": launch_response["evaluations"][0].get("eval_subtype"),
        }
        return self.evaluation_content

    def get_status(self, eval_id: str = None):
        if eval_id is not None:
            eval_result = check_eval_result(self.url, eval_id)
        else:
            eval_result = check_eval_result(self.url, self.evaluation_content.get("id"))
        return eval_result['status']

    def get_results(self, eval_id: str = None):
        if eval_id is None:
            eval_result = check_eval_result(self.url, self.evaluation_content.get("id"))
        eval_result = check_eval_result(self.url, eval_id)

        # ret = []
        return eval_result
        # for e in eval_results:
        #     results = e['aggregated_results']
        #     for res in results:
        #         metric_name = res['name']
        #         metric_value = f"{res['value'] * 100:.2f}"
        #         ret.append({metric_name: metric_value})

        # return ret
#
#
# repo_name = "yangj-dataset-test-llm-judge"
#
# suffix = '-' + str(uuid.uuid4())
# repo_name += suffix
# print ("repo_name", repo_name)
#
# repo_full_name = f"nvidia/{repo_name}"
# print ("repo_full_name", repo_name)
#
# dir_path = "/app/yangj/Eval-MS/eval_dataset_llm_judge_test3"
# print ("dir_path", dir_path)
#
# params = llm_params = {
#    "model": {
#           "llm_type": "nvidia-nemo-nim",
#           "llm_name": "nv-gpt-8b-base",
#           "inference_url": "http://gpt-8b-nemollm-inference.nemo-evaluation.svc.cluster.local:8006/v1",
#         # "inference_url": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/bc205f8e-1740-40df-8d32-c4321763498a"
#           "is_chat_model": False
#     },
#     "evaluations": [
#         {
#             "eval_type": "llm_as_a_judge",
#             "eval_subtype": "mtbench",
#             "bench_name": "not_mt_bench",
#             "mode": "single",
#             "input_dir": f"nds:{repo_name}",
#
#             "inference_params": {
#                 "top_p": 0.9,
#                 "top_k": 40,
#                 "temperature": 0.75,
#                 "stop": [],
#                 "tokens_to_generate": 1024
#             },
#
#             "judge_model": {
#                 "llm_type": "nvidia-nemo-nim",
#                 "llm_name": "nv-gpt-8b-base",
#                 "inference_url": "http://gpt-8b-nemollm-inference.nemo-evaluation.svc.cluster.local:8006/v1",
#                 "is_chat_model": False,
#             },
#
#             "judge_inference_params": {
#                 "top_p": 0.9,
#                 "top_k": 40,
#                 "temperature": 0.5,
#                 "stop": [],
#                 "tokens_to_generate": 1024,
#             }
#         }
#     ],
#     "tag": repo_name,
# }
#
# DATASET_ID = None
# DATASET_ID = create_dataset(repo_name)
# print ("created dataset id", DATASET_ID)
#
# upload_result = upload_dir(dir_path, repo_full_name)
#
# endpoint = f"{EVAL_URL}/v1/evaluations"
# evaluation_response = requests.post(endpoint, json=llm_params).json()
#
# print ("evaluation_response", evaluation_response)
#
# EVAL_ID = evaluation_response["evaluation_id"]
# print ("Eval response", evaluation_response)
# print ("Eval id: ", EVAL_ID)
#
# eval_result = check_eval_result(EVAL_ID)
# status = eval_result['status']
# print ("Eval evaluation_id: ", EVAL_ID)
# print ("Eval result status: ", status)
#
# if status == 'succeeded':
#     tag = eval_result['tag']
#     model = eval_result['model']
#     metrics = eval_result['evaluation_results']
#     print ("Eval result tag: ", tag)
#     print ("Eval result metrics: \n", metrics)
#
#
#
# evaluation_full_path = f"nvidia/{EVAL_ID}"
# print ("result_repo_full_name", EVAL_ID)
#
# download_results_to_local_directory(evaluation_full_path, f"my_eval_{dir_path.split('/')[-1]}")
