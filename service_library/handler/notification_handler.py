import json
import logging
from io import BytesIO
import os
from typing import List

import pandas as pd
import boto3
from fastapi import UploadFile, File

from configs.settings import get_settings
from constants import common

from service_library.constants import NT_ACCOUNT_ID, NT_ACCOUNT_NAME
from service_library.notification.send_email import EmailServices
from service_library.utils.data_helper import byte_content_convertor
from service_library.utils.logging import log_errors
from service_library.utils.request_helpers import create_header
from datadog import initialize, statsd, api

logger = logging.getLogger(__name__)


class NotificationHandler:

    def __init__(self, config: dict = None):
        if config is None:
            config = dict()

        self.config = config

    @log_errors("Send Notification")
    def send_notification(self, subject: str, email_recipients: List,  notification_data: dict):
        if notification_data:
            try:
                logging.info(f"email recipients: {email_recipients}")
                email_service_response = EmailServices().send_email(
                        subject,
                        email_recipients,
                        notification_data
                    )
                logging.info(f"email_service_response: {email_service_response}")
                return {"notification": email_service_response}
            except Exception as ex:
                logging.error(f"error when sending email notification: {ex}")