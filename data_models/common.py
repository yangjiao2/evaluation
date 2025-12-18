from typing import Dict, List, Optional, Any

from nvbot_utilities.constants.bot_configuration import PlatformAgentSelection, PlatformModelSelection
from pydantic import BaseModel, Json

from nvbot_models.nvbot_platform_schemas.contracts.config_contracts.langchain_config.nv_langchain_config import \
    NVBotPlatformConfig

#
# class UserInfo(BaseModel):
#     # ExecutionId: Optional[str]
#     NvidiaID: Optional[str]
#     Username: Optional[str]


class EvaluationRequest(BaseModel):
    Project: str
    Runtype: Optional[str] = 'manual'
    Metadata: Optional[dict]
    # Userinfo: Optional[UserInfo]

    SourceSystem: Optional[PlatformAgentSelection]
    Model: Optional[PlatformModelSelection]
    PlatformConfig: Optional[NVBotPlatformConfig]

    class Config:
        use_enum_values = True


class EvaluationResponse(BaseModel):
    # Response: Response
    StatusCode: int
    Status: str = ""
    Details: Dict[str, Any] = {}
    Error: Optional[Any]


class RunResult(BaseModel):
    Status: str
    Latency: float
    Annotation: Optional[str]
    CreatedDateTime: Optional[str]


class ExtractOutputKeys(BaseModel):
    Type: str
    Name: str
    MapTo: Optional[str]
