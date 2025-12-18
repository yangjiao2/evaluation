import json
import logging
from io import BytesIO
import os
from os import environ
from typing import List

from fastapi import UploadFile, File

from configs.settings import get_settings
from constants import common

from service_library.constants import NT_ACCOUNT_ID, NT_ACCOUNT_NAME
from service_library.utils.data_helper import byte_content_convertor
from service_library.utils.logging import log_errors
from service_library.utils.request_helpers import create_header
from datadog import statsd, initialize

logger = logging.getLogger(__name__)



# initialize datadog to talk to our nvbot datadog monitoring service
options = {
    "statsd_host": get_settings().DATADOG_HOST, # "datadog-monitoring.default.svc.cluster.local",
    "statsd_port": 8125,
}
initialize(**options)


class DatadogHandler:

    def __init__(self, config: dict = None):
        if config is None:
            config = dict()

        self.config = config


    # def initialize():
    # """
    # Initialize Datadog APIs.
    #
    # To enable Datadog APIs, the following keys should be set in OS environment:
    #     - DATADOG_API_KEY(or DD_API_KEY)
    #     - DATADOG_APP_KEY(or DD_APP_KEY)
    #     - DATADOG_STATSD_HOST(or DD_STATSD_HOST)
    # """
    #
    #     logger.info('Initializing Datadog...')
    #     api_key = environ.get("DATADOG_API_KEY", environ.get("DD_API_KEY"))
    #     app_key = environ.get("DATADOG_APP_KEY", environ.get("DD_APP_KEY"))
    #     statsd_host = environ.get('DATADOG_STATSD_HOST',
    #                               environ.get('DD_STATSD_HOST'))
    #     if api_key and app_key:
    #         try:
    #             dd_initialize(api_key=api_key,
    #                           app_key=app_key,
    #                           statsd_host=statsd_host)
    #         except Exception as ex:  # pylint: disable=broad-except
    #             logger.error('Failed initializing datadog: %s', ex)
    #     else:
    #         logger.warning('No api_key and app_key are set. Datadog is disabled. '
    #                        'Please set environment DATADOG_API_KEY and '
    #                        'DATADOG_APP_KEY to enable Datadog.')

    @log_errors("Insert metric to Datadog")
    def send_eval_metrics(self, eval_result: dict, metrics: dict, tags: List[str]):
        project = eval_result.get("project_name", "project_name")
        eval_id = eval_result.get("run_id", "run_id")
        bot_name = eval_result.get("bot_name", "bot_name")
        logger.info(f"Sending {bot_name} metrics to Datadog: {eval_id}")

        bot_name = eval_result.get("bot_name", "bot_name")
        # timestamp = eval_result['timestamp']      # let gauge handles this
        print (f"Eval metrics send to datadog: {metrics}")
        logger.info(f"Eval metrics send to datadog: {metrics}")
        for metric, value in metrics.items():
            # Sending as a GAUGE
            statsd.gauge(f"eval.{metric}", value, tags)
            # Sending as a HISTOGRAM
            statsd.histogram(f"eval.{metric}_distribution", value, tags)

#
# current_span = get_current_span()
# datadog_span_tags = set_span_tags(current_span=current_span,
#                                   span_tag_key="Model_Name",  # Make from constatns TODO sean
#                                   span_tag_value=target_model)
