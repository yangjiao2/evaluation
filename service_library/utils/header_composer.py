from controllers.auth.auth_token_loader import AuthTokenLoader
import os
import re
import logging
from service_library.utils.logging import log_errors
logger = logging.getLogger(__name__)


@log_errors("Header auth composer")
def header_auth_composer(auth_config, row: dict = None):
    """
    Composes authentication based on the provided auth configuration.

    Supports:
    - Basic Auth (username, password)
    - Bearer Token Auth (token)
    """

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    if not auth_config or not auth_config.get("AuthType"):
        return headers, None

    auth_type = auth_config.get("AuthType", "basic").lower()
    auth = None
    if auth_type == "basic":
        username = row.get("Auth.Username") if (row is not None and row.get("Auth.Username")) else auth_config.get("Username")
        auth = (username, auth_config["Password"])
    elif auth_type == "bearer":
        auth = {"Authorization": f"Bearer {auth_config['Token']}"}

        headers.update(auth)
        auth = None  # Set auth to None since we're passing it in headers
    elif auth_type == "starfleet":
        headers.update({"Authorization": f"Bearer {AuthTokenLoader(auth_config.get('env', 'prd')).token}"})
    # example: auth_config
    # {
    #     "AuthType": "custom",
    #
    #     "Payload": {
    #         "x-auth-mode": "nvauth",
    #         "x-nvauth-system-account-token": "$IT_SUPPORT_NVAUTH_ACCOUNT_TOKEN$",
    #         "Authorization": "Bearer $IT_SUPPORT_BEARER_TOKEN$"
    #     }
    # }

    elif auth_type == "custom":
        payload = auth_config.get("Payload", {})
        processed_payload = {}
        
        for key, value in payload.items():
            if isinstance(value, str):
                # Find all environment variable placeholders in the format $VAR_NAME$
                env_vars = re.findall(r'\$([A-Z_][A-Z0-9_]*)\$', value)
                processed_value = value
                for env_var in env_vars:
                    env_value = os.getenv(env_var)
                    if env_value is None:
                        logger.error(f"Environment variable '{env_var}' not found")
                        print (f"Environment variable '{env_var}' not found")
                    else:
                        print(f"Environment variable loaded '{env_var}'='{env_value}'")
                    
                    processed_value = processed_value.replace(f'${env_var}$', env_value)
                
                processed_payload[key] = processed_value
            else:
                processed_payload[key] = value
        # print (f"Custom header: {processed_payload}")
        headers.update(processed_payload)


    return headers, auth
