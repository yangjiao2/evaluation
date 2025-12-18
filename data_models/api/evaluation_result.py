# flake8: noqa
from __future__ import annotations

from dataclasses import Field, dataclass

from dataclasses_json import dataclass_json
from fastapi import Query
from pydantic import BaseModel, ConfigDict

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)
# from uuid import UUID


from nvbot_models.nvbot_platform_schemas.contracts.config_contracts.langchain_config.nv_langchain_config import \
    NVBotPlatformConfig
from nvbot_models.request_models.bot_maker_request import BotmakerRequestParameter, BotMakerRequest
from nvbot_models.request_models.fulfillment_request import Attachment

from pydantic import BaseModel  # EmailStr
from typing import List, Dict, Optional, Union


# @dataclass
class RegressionRunMetrics(BaseModel):
    Processed: Optional[int] = 0
    Total: Optional[int] = 0
    Avg: Optional[float] = None
    Std: Optional[float] = None
    P50: Optional[float] = None
    P90: Optional[float] = None
    P95: Optional[float] = None
    P99: Optional[float] = None

    class Config:
        allow_extra = False


class EvalRunMetrics(BaseModel):
    RunId: Optional[str] = None
    ProjectId: Optional[str] = None
    ProjectName: Optional[str] = None
    BotName: Optional[str] = None
    Metrics: Dict[str, float | int] = None

    # common
    class Config:
        allow_extra = True


