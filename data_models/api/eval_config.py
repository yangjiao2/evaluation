# flake8: noqa
from __future__ import annotations

from dataclasses import Field, dataclass

from dataclasses_json import dataclass_json
from pydantic import BaseModel

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


class ConfigResponse(BaseModel):
    evaluation_config: Optional[dict] = None
    details: Optional[str] = None
