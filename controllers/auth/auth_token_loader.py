import json
import os

import requests
import time
import logging

from authlib.jose import jwt

from configs.settings import get_settings, get_cache_session

from service_library.utils.data_helper import safe_json_loads
from service_library.utils.logging import log_errors

logger = logging.getLogger(__name__)

# device flow guide: https://gitlab-master.nvidia.com/kaizen/auth/starfleet/sf-docs/-/blob/main/apis/user-authentication/device-authorization-endpoint.md
# token guide: https://gitlab-master.nvidia.com/kaizen/auth/starfleet/sf-docs/-/blob/main/apis/user-authentication/token-endpoint.md#poll-the-status-of-an-off-device-login
# refresh token: https://gitlab-master.nvidia.com/kaizen/auth/starfleet/sf-docs/-/blob/main/overview/extras/refresh-tokens.md?ref_type=heads

class AuthTokenLoader:
    def __init__(self, env: str = "prd"):
        _settings = get_settings()
        self.auth_env = "prd"  if env.lower() == "prd" else "nonprd"
        print ("auth_env", self.auth_env)

        self._build_secrets()

        self.cache_session = get_cache_session()
        logger.info(f"loading cache from {self.cache_session.redis_url}")

        self.auth_token = None
        self.expires_at = None
        self.refresh_token = None

        # for refresh token flow
        self.refresh_token = self.cache_session.get_from_cache(key='refresh_token', key_hash=f"{self.client_id}:refresh_token")

        # for client_token flow
        self.sub = self.cache_session.get_from_cache(key='sub', key_hash=f"{self.client_id}:sub")
        self.client_token = self._get_client_token()
        logger.info(f"Token configuration loaded successfully for {self.auth_env} env")

    @log_errors('getting client token')
    def _get_client_token(self):
        client_token = None
        key_hash = f"{self.client_id}:client_token"
        # print ("key_hash", key_hash)
        # print ("token", key_hash)
        if key_hash:
            client_token = self.cache_session.get_from_cache(key="client_token", key_hash=key_hash)
            # logger.info(f"loaded client token {client_token} for {self.client_id} from cache")
            # print(f"Loaded client token for {client_token}  for {self.client_id} from cache")
            if not client_token:
                # TODO: renew token
                print ("Please renew token via generate_client_token flow")
                logger.warning(f"Please renew token via generate_client_token flow")
                # self.generate_client_token(self.client_id)
                return self.cache_session.get_from_cache(key="client_token", key_hash=key_hash)
        return client_token


    @property
    def expired_at(self):
        return self.expires_at


    @log_errors('generate auth token')
    def fetch_auth_token(self):
        auth_token = None
        client_token = self._get_client_token()
        logger.info(f"Fetching auth token by client_token {client_token} to {self.token_url}")
        print (f"Fetching auth token by client_token {client_token} towards {self.token_url}")
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {
            "client_token": client_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:client_token",
            "client_id": self.client_id,
            "sub": self.sub,
        }
        response = requests.post(self.token_url, headers=headers, data=data)
        if response.status_code == 200:
            auth_token_response_dict = response.json()
            auth_token = auth_token_response_dict["id_token"]
            self.expires_at = auth_token_response_dict["expires_in"]
            self.auth_token = auth_token
            logger.info(f"Fetched auth token, expires in {self.expires_at}")

            key_hash = f"{self.client_id}:auth_token"
            self.insert_to_cache(key="auth_token", value=self.auth_token, key_cache=key_hash)
            return auth_token
        message = f"Failed to fetched auth token from {self.token_url}, status code {response.status_code}, {response.text}"
        logger.warning(message)
        print (message)
        print (f"Auth request payload: \n {data}")
        assert auth_token, message
        return None

    @log_errors('generate auth token')
    def fetch_auth_token_by_refresh_token(self):
        auth_token = None
        logger.info(f"Fetching auth token by refresh_token {self.refresh_token} to {self.token_url}")
        print (f"Fetching auth token by refresh_token {self.refresh_token} towards {self.token_url}")
        client_secret_key_hash = f"{self.client_id}:client_secret"
        client_secret = self.cache_session.get_from_cache(key='client_secret', key_hash=client_secret_key_hash)
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {
            "refresh_token": self.refresh_token,
            "grant_type": "refresh_token",
            'client_id': self.client_id,
            'client_secret': client_secret,
        }
        response = requests.post(self.token_url, headers=headers, data=data)

        if response.status_code == 200:
            auth_token_response_dict = response.json()
            auth_token = auth_token_response_dict["id_token"]
            self.expires_at = auth_token_response_dict["expires_in"]
            self.auth_token = auth_token
            self.refresh_token = auth_token_response_dict["refresh_token"]

            logger.info(f"Fetched auth token, expires in {self.expires_at}, with refresh token {self.refresh_token}")
            refresh_key_hash = f"{self.client_id}:refresh_token"
            self.insert_to_cache(key="refresh_token", value=self.refresh_token, key_cache=refresh_key_hash, expires_in=604800) # 7 day

            key_hash = f"{self.client_id}:auth_token"
            self.insert_to_cache(key="auth_token", value=self.auth_token, key_cache=key_hash)
            return auth_token
        message = f"Failed to fetched auth token from {self.token_url}, status code {response.status_code}, {response.text}"
        logger.warning(message)
        print (message)
        print (f"Auth request payload: \n {data}")
        assert auth_token, message
        return None


    @property
    def token(self):
        key_hash=f"{self.client_id}:auth_token"
        # logger.info(f"Fetch auth_token to cache with key {key_hash}")
        # print((f"Fetch auth_token to cache with key {key_hash}"))
        self.auth_token = self.cache_session.get_from_cache(key='auth_token', key_hash=key_hash)
        if self.auth_token:
            # print ("token", self.auth_token)
            # assert self.auth_token is not None, "Not able to get Auth token from cache"
            return self.auth_token

        if self.refresh_token:
            self.auth_token = self.fetch_auth_token_by_refresh_token()
        if self.auth_token is None:
            if self.refresh_token:
                print(f"\n⚠️ auth_token is None via refresh token flow`{self.refresh_token}`")
                logger.error(f"\n⚠️ auth_token is None via refresh token flow`{self.refresh_token}` with client_id {self.client_id}`")

            self.auth_token = self.fetch_auth_token()
            if self.auth_token is None:
                print(f"\n⚠️ auth_token is None via client id token flow`{self.refresh_token}`")
                logger.error(f"\n⚠️ auth_token is None via client id token token flow`{self.refresh_token}` with client_id {self.client_id}")
        assert self.auth_token is not None, "Not able to get Auth token from starfleet token validation, please contact admin of nvbot cli starfleet. Sorry!"
        # print ("\n⚠️ auth_token: ", self.auth_token)
        return self.auth_token

    @log_errors('insert token to cache')
    def insert_to_cache(self, key: str, value: str, key_cache: str = None, expire_in: int = 3600):
        if not key_cache:
            key_cache = f"{self.client_id}:{key}"
        logger.info(f"Insert {key} token to cache with key {key_cache}")

        self.cache_session.update_expire_cache(
            key=key,
            value=value,
            exp_time=expire_in,
            key_hash=key_cache
        )
        return value


    @log_errors('generate client token')
    def _generate_client_token(self, client_id):
        device_id = 'thisisadeviceid'  # random
        display_name = 'deviceFlow'
        scope = 'openid email profile'

        # Get `device_code`, and URI
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {
            'client_id': client_id,
            'device_id': device_id,
            'display_name': display_name,
            'scope': scope
        }

        response = requests.post(self.authorize_url, headers=headers, data=data)
        assert response.status_code == 200
        response_dict = json.loads(response.text)
        device_code = response_dict["device_code"]
        go_to_uri = response_dict["verification_uri_complete"]
        print("\nGo to this url, click 'submit', 'continue': \n", go_to_uri)

        # Get `id token`, and `sub`
        data = {
            'device_code': device_code,
            'grant_type': 'urn:ietf:params:oauth:grant-type:device_code',
            'client_id': client_id,
        }
        response = requests.post(self.token_url, headers=headers, data=data)
        assert response.status_code == 200
        token_response_dict = json.loads(response.text)
        id_token = token_response_dict["id_token"]
        print("\nGet `sub` from id_token: \n", id_token)
        access_token = token_response_dict["access_token"]
        expires_in = token_response_dict["expires_in"]

        # Get client token
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {access_token}"

        }
        response = requests.get(self.client_token_url, headers=headers)
        assert response.status_code == 200
        token_response_dict = json.loads(response.text)
        client_token_response_dict = json.loads(response.text)
        client_token = client_token_response_dict['client_token']

        client_token_response_dict = json.loads(response.text)
        client_token = client_token_response_dict['client_token']
        client_token_expires_in = client_token_response_dict["expires_in"]
        # client token expect to expire in 90 days
        print(f"client token {client_token}, expires in {client_token_expires_in} secs")
        logging.info(f"Fetched client token, expires in {client_token_expires_in}")

        self.insert_to_cache(key="auth_token",
                             value=self.auth_token,
                             key_cache=client_id,
                             expire_in=client_token_expires_in
                             )
        return client_token, client_token_expires_in


    def _get_sub_from_token(self, jwt_token):
        decoded_token = jwt.decode(jwt_token, options={"verify_signature": False})

        sub_claim = decoded_token.get("sub")
        return sub_claim
    

    def _build_secrets(self):
        _settings = get_settings()
        assert self.auth_env, "Expect auth_env to be defined"
        self.token_url = _settings.NVSF_TOKEN_URLS[self.auth_env]
        self.authorize_url = _settings.NVSF_AUTHORIZE_URLS[self.auth_env]
        self.client_token_url = _settings.NVSF_CLIENT_TOKEN_URLS[self.auth_env]
        nvsf_client_ids = {
            "prd": os.getenv("NVSF_CLIENT_ID"),
            "nonprd": os.getenv("NVSF_CLIENT_ID_NONPRD") 
        }

        self.client_id = nvsf_client_ids[self.auth_env] if os.getenv("NVSF_CLIENT_ID") and os.getenv("NVSF_CLIENT_ID_NONPRD") else _settings.NVSF_CLIENT_IDS[self.auth_env]
        # print ("client_id", self.client_id)