from __future__ import annotations

import subprocess
import logging
import os
from typing import Any, Dict, Optional

from pydantic import Field

from configs.settings import get_settings
from data_models.api.run_maker import RunMakerRequest
from service_library.utils.logging import log_errors
from pandas import DataFrame


logger = logging.getLogger(__name__)

# https://gitlab-master.nvidia.com/prkumbhar/issue-debugging/-/blob/main/17-genai-perf-workflow-mock/genai_perf_loadgen.py#L11
class PerfTestRunContainer:
    """A container to help manage the state of a perf test run."""

    def __init__(self, project, config, env):
        self.settings = get_settings()
        self.project = project
        self.config = config
        self.env = env

    @log_errors('Performance test generation')
    async def arun(
            self,
            request: RunMakerRequest,
            config: dict,
            dataset: Any,
            **kwargs) -> None:
        
        logger.info("Starting performance test run")
        
        # Extract parameters from the request and config
        model = self.config.PerfTestSchema.Model
        service_kind = self.config.PerfTestSchema.ServiceKind
        endpoint_type = self.config.PerfTestSchema.EndpointType
        tokenizer = self.config.PerfTestSchema.Tokenizer
        url = self.config.PerfTestSchema.URL
        api_key = os.environ.get("API_KEY")  # Or get from a secure location
        concurrency = self.config.PerfTestSchema.Concurrency
        request_count = self.config.PerfTestSchema.RequestCount
        streaming = self.config.PerfTestSchema.Streaming

        if not api_key:
            raise ValueError("API_KEY environment variable not set.")

        # Construct the genai-perf command
        command = [
            "genai-perf", "profile",
            "-m", model,
            "--service-kind", service_kind,
            "--endpoint-type", endpoint_type,
            "--tokenizer", tokenizer,
            "-u", url,
            "-H", f"Authorization: Bearer {api_key}",
            "--concurrency", str(concurrency),
            "--request-count", str(request_count),
            "--output", output_format,
        ]

        if streaming:
            command.append("--streaming")

        logger.info(f"Executing command: {' '.join(command)}")

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    logger.info(output.strip())
            
            rc = process.poll()
            if rc == 0:
                logger.info("genai-perf completed successfully.")
            else:
                logger.error(f"genai-perf failed with exit code {rc}.")

        except FileNotFoundError:
            logger.error("Error: 'genai-perf' command not found.")
            logger.error("Please ensure that the genai-perf library is installed and in your PATH.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    def finish(self, batch_results: list, verbose: bool = False) -> None:
        logger.info("Performance test run finished.")
        return None

    def prepare(
            self,
            request: RunMakerRequest,
            config: dict,
            project_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[DataFrame]:
        logger.info("Preparing for performance test run.")
        return None
    
