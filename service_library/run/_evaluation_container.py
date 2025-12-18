from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Callable

import pandas as pd
from tqdm import tqdm

from data_models.api.run_maker import RunMakerRequest
# from nv_platform.nvbot_platform import NVBotPlatform
from service_library.run.run_container import RunContainer

logger = logging.getLogger(__name__)


class EvaluationRunContainer(RunContainer):
    """A container to help manage the state of a eval run."""

    def __init__(self, project):
        logging.info("Running")
        self.project = project
        self.input_mapper = None
        self.output_mapper = None


    @classmethod
    async def arun(
            cls,
            request: RunMakerRequest,
            config: dict,
            # inputs: dict,
            **kwargs):
        container = EvaluationRunContainer.prepare(
            **kwargs
        )

        df = cls.inputs

        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            search_query = row["Query"]

        try:
            pass
        #     with requests.post(url, headers=headers, json=request.dict()) as response:
        #         if response.status_code == 200:
        #             resp_json = response.json()
        #             df.loc[index, "dump"] = str(resp_json)
        #             df.loc[index, "Text"] = str(resp_json.get("Response", {}).get("Text", ""))
        #             agent_action = str(resp_json.get("Response", {}).get("CleanedText", {}))
        #             df.loc[index, "AgentAction"] = agent_action
        #             # action, action_input = extract_agent_action(agent_action)
        #             # df.loc[index, "Json"] = str(resp_json.get("Response", {}).get("Json", {}))
        #             # df.loc[index, "Tool"] = str(action).strip()
        #             # df.loc[index, "Tool Input"] = str(action_input).strip()
        #             df.loc[index, "IRResults"] = str(resp_json.get("CustomData", {}).get("IRResults", {}))
        except Exception as err:
            print(f"{search_query}: {err}")
        return

    # def _collect_results(
    #         self,
    #         batch_results: List[Union[dict, str]],
    # ) -> TestResult:
    #     # all_eval_results, all_runs = self._collect_metrics()
    #     aggregate_feedback = []
    #     eval_results = []
    #     if self.batch_evaluators:
    #         logger.info("Running result.")
    #         # (eval_results, aggregate_feedback) = self._run_batch_evaluators(batch_results)
    #     results = self._merge_test_outputs(batch_results, eval_results, aggregate_feedback)
    #     return TestResult(
    #         results=results,
    #         aggregate_metrics=aggregate_feedback,
    #     )

    def finish(self, batch_results: list, verbose: bool = False):
        results = None
        try:
            results = self._collect_results(batch_results)

            # input_mapper, output_mapper
            # run_inputs = chain.input_keys if isinstance(chain, Chain) else None
            # run_outputs = chain.output_keys if isinstance(chain, Chain) else None
            # run_evaluators = _load_run_evaluators(
            #     evaluation,
            #     run_type, # regression or evaluation
            #     dataset, #
            #     # list(examples[0].outputs) if examples[0].outputs else None,
            #     run_inputs,
            #     run_outputs,
            # )
            #
            # self.client.update_project(
            #     self.project.id, end_time=datetime.now(timezone.utc)
            # )
        except Exception as ex:
            logger.debug(f"Failed to close project: {repr(ex)}")
        return results

    @classmethod
    def prepare(
            cls,
            request: RunMakerRequest,
            *,
            dataset: Any = None,
            # project: Optional[str],
            input_mapper: Optional[Callable[[Any], Any]] = None,
            concurrency_level: int = 5,
            project_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[RunContainer]:

        # input_mapper, output_mapper
        # run_inputs = chain.input_keys if isinstance(chain, Chain) else None
        # run_outputs = chain.output_keys if isinstance(chain, Chain) else None
        # run_evaluators = _load_run_evaluators(
        #     evaluation,
        #     run_type, # regression or evaluation
        #     dataset, #
        #     # list(examples[0].outputs) if examples[0].outputs else None,
        #     run_inputs,
        #     run_outputs,
        # )

        # fetch dataset
        # get pd data

        # df = S3DatasetHandler(request.Project).read_file_as_pd('nvbot-evaluation', 'nvbot_for_nvhelp_mixtral_agent/dataset/default_evaluation.xlsx')
        df = pd.read_excel('script/sanity.xlsx')

        inputs_columns = input_mapper(df.columns.tolist()) if input_mapper else ['Query']
        cls.inputs = df[inputs_columns]

        return cls

        #
        # return cls(
        #     # client=client,
        #     project=project,
        #     wrapped_model=wrapped_model,
        #     examples=examples,
        #     configs=configs,
        #     batch_evaluators=evaluation.batch_evaluators if evaluation else None,
        # )
