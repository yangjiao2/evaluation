# flake8: noqa
from __future__ import annotations

from dataclasses import Field, dataclass

import pandas as pd
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
    runtime_checkable, TypedDict, Literal,
)
# from uuid import UUID

from nvbot_models.nvbot_platform_schemas.contracts.config_contracts.langchain_config.nv_langchain_config import \
    NVBotPlatformConfig
from nvbot_models.request_models.bot_maker_request import BotmakerRequestParameter, BotMakerRequest
from nvbot_models.request_models.fulfillment_request import Attachment
from nvbot_models.nvbot_platform_schemas.contracts.config_contracts.langchain_config.nvbot_langchain_config import (
    NVBotFlowConfig)

from pydantic import BaseModel  # EmailStr
from typing import List, Dict, Optional, Union


class StorageType(Enum):
    S3 = "s3"
    DATASTORE = "datastore"
    LOCAL = "local"
    DATABASE = "database"
    DATAFRAME = "dataframe"


class DatasetConfig(BaseModel):
    # common
    Engine: Optional[str] = None  # check StorageType
    DataLimit: Optional[int] = None
    DataShuffleSeed: Optional[int] = None
    Filters: Optional[Dict[str, str]] = None

    # upload to s3, download from s3
    Name: Optional[str] = None
    ResultFolder: Optional[str] = None
    DatasetPath: Optional[str] = None

    # local
    RunFile: Optional[str] = None

    # dataframe
    Data: Optional[Any] = None

    # datastore
    DatasetId: Optional[str] = None
    Files: Optional[List[str]] = []
    DatasetFolder: Optional[str] = None

    # database
    HistoryId: Optional[str] = None


class JudgeConfig(BaseModel):
    Prompt: Optional[str] = None
    Inputs: Optional[dict] = None


class Metrics(BaseModel):
    name: Optional[str] = None
    params: Optional[List[int]] = None
    score_type: Optional[str] = None
    threshold: Optional[int] = None


class NemoEvaluator(BaseModel):
    # Inputs: Optional[dict]
    Evaluators: List[dict] = None
    DatasetConfig: Optional["DatasetConfig"] = DatasetConfig


class CustomEvaluator(BaseModel):
    # Assuming all fields in CustomEvaluator are also optional
    Evaluators: List[dict] = None
    DatasetConfig: Optional["DatasetConfig"] = None


class Notification(BaseModel):
    EmailRecipients: Optional[List[str]] = []


class EvaluationSchema(BaseModel):
    # Id: Optional[str]
    Notification: Optional[Notification] = Notification
    NemoEvaluator: Optional[NemoEvaluator] = NemoEvaluator


class RunParamConfig(BaseModel):
    name: Optional[str] = None
    type: Optional[Literal["Text", "Attribute", "Function"]] = None
    value: Optional[str] = None
    args: Optional[list] = []


class RunConfig(BaseModel):
    Inputs: Optional[List[RunParamConfig]] = []
    Outputs: Optional[List[RunParamConfig]] = []


class RegressionSchema(BaseModel):
    # Id: Optional[str]
    Notification: Optional[Notification] = Notification
    RunConfig: Optional[RunConfig] = RunConfig
    DatasetConfig: Optional[DatasetConfig] = DatasetConfig
    DataConfigs: Optional[dict] = None


class ComparisonSchema(BaseModel):
    # Id: Optional[str]
    Notification: Optional[Notification] = None
    DataConfigs: Optional[dict] = None
    DatasetConfigs: Optional[List[DatasetConfig]] = []
    Comparator: Optional[List[dict]] = None
    Outlier: Optional[dict] = None


class NVBotEvaluationConfig(BaseModel):
    EvaluationSchema: Optional["EvaluationSchema"] = EvaluationSchema
    RegressionSchema: Optional["RegressionSchema"] = RegressionSchema
    ComparisonSchema: Optional["ComparisonSchema"] = ComparisonSchema


class FlowConfigRequest(BaseModel):
    ConfigId: Optional[int] = None
    System: Optional[str] = None
    Model: Optional[str] = None  # Optional[PlatformModelSelection]
    UserId: Optional[str] = None
    Env: Optional[str] = "stg"


class APIConfigRequest(BaseModel):
    ConfigType: Optional[str] = "api"
    URL: Optional[str] = None
    Auth: Optional[dict] = None
    Header: Optional[dict] = None
    Payload: Optional[dict] = None
    Type: Optional[str] = "POST"


class BotPlatformConfig(BaseModel):
    ConfigType: Optional[str] = "platform"
    FlowConfig: Optional[NVBotFlowConfig] = None


## RunMakerRequest
class RunMakerRequest(FlowConfigRequest):
    PlatformConfig: Optional[Union[APIConfigRequest | BotPlatformConfig]] = None
    Project: Optional[str] = ""
    ProjectId: Optional[int] = None
    RunType: Optional[str] = ""
    UserId: Optional[str] = ""
    UserName: Optional[str] = ""
    Parameters: Optional[dict] = None
    EvaluationConfig: Optional[NVBotEvaluationConfig] = None

    Customization: Optional[dict] = None


class EvaluationProcessingRequest(BaseModel):
    Project: Optional[str] = None
    ProjectId: Optional[int] = None
    RunType: Optional[str] = None
    CreatedDateFrom: Optional[str] = None
    CreatedDateTo: Optional[str] = None
    ModifiedDateFrom: Optional[str] = None
    ModifiedDateTo: Optional[str] = None
    Tag1: Optional[str] = None
    Tag2: Optional[str] = None
    IsActive: Optional[bool] = True
    Status: Optional[str] = Query("STARTED",
                                  description="Refers to evaluation run status: STARTED, IN_PROCESS, COMPLETED, FAILED")

# class MetricsProcessingRequest(BaseModel):
#         HistoryIds: Optional[list] = None

class ColumnMapModel(BaseModel):
    question: str = "Query"
    reference: str = "Answer"
    additional_params: List[str] = None


class JudgeModel(BaseModel):
    prompt_module: str = None
    output_format: str = None
    template: str = None
    scorers: List[str] = None
