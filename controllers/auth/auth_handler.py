import logging

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader


from nvbot_utilities.utils.auth.auth_handler import AuthToken
from configs.settings import get_settings

# Header key that needs to be used for auth
api_key_header = APIKeyHeader(name='API_KEY', auto_error=False)
log = logging.getLogger('AuthHandler')
app_settings = get_settings()


async def get_api_key(api_key: str = Security(api_key_header)):
    key = app_settings.API_KEY
    if api_key != key:
        raise HTTPException(status_code=401, detail="Invalid token")


async def check_auth(request: Request,
                     token=Depends(AuthToken(feedback_secrets=app_settings.STARFLEET_SECRETS,
                                             redis_secrets=app_settings.REDIS_SECRETS,
                                             allowed_dl_list=app_settings.ALLOWED_DL_LIST,
                                             helios_api_key=app_settings.HELIOS_API_KEY,
                                             settings=settings).verify_token)):
    if not token:
        log.error('Missing both bearer and token auth')
        raise HTTPException(detail='User not authenticated', status_code=401)
