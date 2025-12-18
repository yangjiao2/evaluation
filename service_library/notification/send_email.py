import logging
import os
import uuid
from pathlib import Path
from typing import Union, List, Any

from nvbot_models.utils.send_grid import SendGrid
from jinja2 import Environment, FileSystemLoader
from sendgrid import SendGridAPIClient, Mail

from configs.settings import get_settings, get_cache_session

import requests

from nvbot_models.response_models.bot_maker_fulfillment_response import BotMakerFulfillmentResponse
from nvbot_utilities.utils.starfleet.device_token import DeviceToken
from nvbot_utilities.utils.starfleet.starfleet_models import StarfleetConfig
from service_library.utils.logging import log_errors

log = logging.getLogger('EmailServices')


class EmailServices:
    def __init__(self):
        _settings = get_settings()
        self.sendgrid_token = _settings.SENDGRID_TOKEN
        self.cache_session = get_cache_session()
        base_dir = Path(__file__).resolve().parent
        self.template_dir = base_dir.joinpath('templates')
        self.env = Environment(loader=FileSystemLoader(self.template_dir))

    def get_services_token(self, username: str, query_id: str):
        feedback_secrets = StarfleetConfig.model_validate(get_settings()['feedback'])
        device_token = DeviceToken(self.cache_session, feedback_secrets)
        token = device_token.get_services_token(username, query_id, 'feedback')
        return token

    # def get_headers(self, username: str, query_id: str = str(uuid.uuid4())):
    #     token = self.get_services_token(username, query_id)
    #     headers = {
    #         'Content-Type': 'application/json',
    #         'Authorization': f'Bearer {token}',
    #         'device-initiated': 'true',
    #         'query_id': query_id
    #     }
    #
    #     return headers

    def _mail_render(self, data, template) -> str:
        log.debug('Rendering template %s', template)
        text = self.env.get_template(template)
        msg = text.render(data)
        return msg

    @log_errors("Send emails")
    def send_email(self, subject: str, recipients: Union[str, List[str]], data: Any, template: str = 'run-report.html', plain_text_context: Any = None):
        from_email = "nvbot-evaluation@exchange.nvidia.com"
        delivery_status = {"success": [], "failure": []}
        # recipients = ["yangj@nvidia.com"]
        # Send the email
        for recipient in recipients:
            try:
                status = False
                # SendGrid(token=self.sendgrid_token).send_email(
                #     data=data,
                #     from_user='NVBot Evaluation',
                #     from_email=from_email,
                #     recipient=recipient,
                #     subject=subject,
                #    ) #  template=template
                html_content = self._mail_render(data, template) if template else None
                plain_text_context = data if not template else None
                try:
                    sendgrid = SendGridAPIClient(self.sendgrid_token)
                    message = Mail(
                    from_email=f'NVBot Evaluation <{from_email}>',
                    to_emails=recipient,
                    subject=subject,
                    html_content=html_content,
                    plain_text_content=plain_text_context)

                    sendgrid.send(message)
                    status = True
                except Exception as ex:
                    log.error('Error when sending email to %s; %s', recipient, ex)
                    status = False
                if status:
                    delivery_status["success"].append(recipient)
                else:
                    delivery_status["failure"].extend(recipient)
                log.info(f"Email status {status} to recipient {recipient}")
                print(f"Email status {status} to recipient {recipient}")
            except Exception as ex:
                log.info(f"Email send to recipient {recipient} failure: {ex}")

        delivery_success_recipients = ', '.join(delivery_status["success"])
        delivery_failure_recipients = ', '.join(delivery_status["failure"])

        logging.info(f"email delivery success: {delivery_success_recipients}")
        logging.info(f"email delivery failure: {delivery_failure_recipients}")
        return {"email_delivery": delivery_success_recipients,
                "email_delivery_failure": delivery_failure_recipients} \
            if delivery_failure_recipients else {"email_delivery": delivery_success_recipients}

        # TODO: use queue service

        # queue_name: str = get_setting().email_queue
        # message_attributes = self.set_message_attributes("Notification")
        # message_body = json.dumps(email.dict())
        # queue_request: QueueRequest = QueueRequest(QueueName=queue_name,
        #                                        MessageBody=message_body,
        #                                        MessageAttributes=message_attributes)
        # queue_response = self.process_request(queue_request)
        # log.info('Prospero Email message posted to queue')
        #
        # except Exception as err:
        # log.error("Failed to sent message.Error: %s", err)
        # queue_response.Message = f"{self.error_message}: {err}"
