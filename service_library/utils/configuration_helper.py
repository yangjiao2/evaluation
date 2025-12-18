import logging
from typing import Optional, Union

from data_models.api.run_maker import RunMakerRequest, EvaluationSchema, APIConfigRequest, BotPlatformConfig
from service_library.handler.config_loader import ConfigLoader
from service_library.handler.database_handler import DatabaseHandler
from service_library.utils.logging import log_errors

from nvbot_models.nvbot_platform_schemas.contracts.config_contracts.langchain_config.nv_langchain_config import \
    NVBotPlatformConfig

@log_errors('Get Platform Config By Type')
def get_platform_config_model_from_dict(platform_config_schema: dict) -> Optional[Union[APIConfigRequest | BotPlatformConfig]]:
    if not platform_config_schema:
        return None

    config_type = platform_config_schema.get('ConfigType').lower() if platform_config_schema.get("ConfigType", "") else ""
    if config_type == "api":
        return APIConfigRequest.model_validate(platform_config_schema)
    else:
        return BotPlatformConfig.model_validate(platform_config_schema)


@log_errors('Get Evaluation Config By Type')
def get_evaluation_config_by_nemo_evaluator_type(evaluation_schema: EvaluationSchema, evaluator_type: str) -> Optional[dict]:
    if evaluation_schema is None or evaluation_schema.NemoEvaluator is None:
        return None
    nemo_evaluator = evaluation_schema.NemoEvaluator
    nemo_evaluators_entries = nemo_evaluator.Evaluators
    print (f"Contains {len(nemo_evaluators_entries)} evaluator entries")
    for entry in nemo_evaluators_entries:
        payload = entry["evaluator_payload"]
        print ("Payload: ", payload)
        # prepare datastore asset
        eval_type = payload.get('eval_type')
        if eval_type.lower() == evaluator_type:
            return entry

    if evaluator_type:
        error_message = f"Empty response got for get evaluation schema by evaluator type: {evaluator_type}"
        print (error_message)
        return None


@log_errors('Get Evaluation Project By Id')
def get_evaluation_project_by_id(project_id: int):
    try:
        if project_id is None:
            return None
        evaluation_projects = DatabaseHandler({'env': 'dev'}).get_evaluation_project(
            {"id": project_id, "is_active": True})
        if evaluation_projects is not None and len(evaluation_projects) > 0:
            evaluation_project = evaluation_projects[0]
            return evaluation_project
    except Exception as ex:
        logging.error(f"Failed to get evaluation project by id: {project_id}, ex")
        error_message = f"Failed to get evaluation project by id {project_id}, ex"
        print (error_message)
        raise Exception(error_message)

@log_errors('Get Platform Config')
async def get_platform_config(run_request: RunMakerRequest, evaluation_project: dict):
    json_platform_config = None
    env = run_request.Env
    if evaluation_project is None:
        evaluation_project = get_evaluation_project_by_id(run_request.ProjectId)
        # if json_platform_config:
        #     print(f"get bot config from ProjectId {run_request.ProjectId}")
        # else:
        #     print(f"Failed to get bot config by request projectId: {run_request.ProjectId}")
    try:
        if evaluation_project and evaluation_project.get("flowconflowconfigfig_metadatajson") is not None:
            json_platform_config = evaluation_project.get("flowconfig_metadatajson")
            if json_platform_config:
                print("get bot config from EvaluationHistory")
            else:
                print(f"Failed to get bot evaluation config from evaluation_project: {evaluation_project.get('id')}")
        else:
            logging.info(f"Loading platformConfig")
            json_platform_config = await ConfigLoader(env).get_flow_model_config(run_request)
            if json_platform_config:
                print(f"get bot config from ConfigId = {run_request.ConfigId}, System={run_request.System} and Model={run_request.Model}")
            else:
                print(f"Failed to get bot config from ConfigId = {run_request.ConfigId}, System={run_request.System} and Model={run_request.Model}")
    except Exception as ex:
        logging.error(f"Failed to get bot flow config from evaluation project, {ex}")
        error_message = f"Failed to get bot flow config from evaluation project, {ex}"
        print (error_message)
        raise Exception(error_message)
    finally:
        if json_platform_config:
            json_platform_config["ConfigType"] = "platform"
        return json_platform_config

@log_errors('Get Evaluation Config')
async def get_evaluation_config(run_request: RunMakerRequest, evaluation_project: dict) -> dict:
    json_eval_config = None
    env = run_request.Env
    if evaluation_project is None:
        evaluation_project = get_evaluation_project_by_id(run_request.ProjectId)

    try:
        if evaluation_project and evaluation_project.get("evaluation_metadatajson", {}) != {}:
            json_eval_config = evaluation_project.get("evaluation_metadatajson")
            if json_eval_config:
                print("get eval config from EvaluationHistory")
        elif run_request.Project:
            # TODO: migrate to config manager repo
            logging.info(f"Load evaluation config based on Project={run_request.Project}")
            json_eval_config = await ConfigLoader().get_evaluation_schema(run_request.Project)
            if json_eval_config:
                print("get eval config from ConfigLoader")

    except Exception as ex:
        logging.error(f"Failed to get evaluation config, {ex}")
        error_message = f"Failed to get evaluation config, {ex}"
        print (error_message)
        raise Exception(error_message)
    return json_eval_config
