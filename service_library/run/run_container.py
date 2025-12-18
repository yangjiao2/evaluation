from __future__ import annotations

import concurrent.futures
import dataclasses
import functools
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union, cast, Callable
from uuid import UUID, uuid4

# from langchain.smith.evaluation.runner_utils import TestResult, _RowResult
from langchain_core.runnables import RunnableConfig

# from langsmith.evaluation import EvaluationResult
from pandas import DataFrame
from pydantic import Field

from configs.settings import get_settings, get_cache_session
from controllers.auth.auth_token_loader import AuthTokenLoader
from data_models.api.run_maker import RunMakerRequest
from service_library.constants import LOCAL_EVAL_RESULTS_TMP_FOLDER
from service_library.utils.logging import log_errors
from service_library.utils.run_helpers import create_folder_if_not_exists

# from nv_platform.nvbot_platform import NVBotPlatform

logger = logging.getLogger(__name__)


class RunContainer:
    """A container to help manage the state of a eval run."""

    def __init__(self, project, config, env):
        self.settings = get_settings()
        self.project = project
        self.config = config
        self.cache_session = get_cache_session()
        self.env = env
        self.auth_token = AuthTokenLoader(env).token

    async def arun(
            cls,
            request: RunMakerRequest,
            config: dict,
            dataset: Any,
            **kwargs):
        return []

    def finish(self, batch_results: list, verbose: bool = False):
        results = None
        return results

    def prepare(
            self,
            request: RunMakerRequest,
            config: dict,
            project_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[DataFrame]:
        return None

    @log_errors("Generate local log output name")
    def _generate_excel_filepath(self, request: RunMakerRequest):
        str_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
        directory = LOCAL_EVAL_RESULTS_TMP_FOLDER
        create_folder_if_not_exists(directory)
        file_path = f"{LOCAL_EVAL_RESULTS_TMP_FOLDER}/{request.Project}-{str_datetime}.xlsx"
        print(f"local log file path: {file_path}")
        logger.info(f"Log saved at: {file_path}")
        return file_path
