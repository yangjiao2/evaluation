# flake8: noqa
from __future__ import annotations

from dataclasses import Field, dataclass

from dataclasses_json import dataclass_json
from fastapi import Header
from fastapi.openapi.models import Reference
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


@dataclass_json
@dataclass
class Dataset(BaseModel):
    """Dataset ORM model."""
    id: str
    name: Optional[str]
    description: Optional[str]
    # last_session_start_time: Optional[datetime] = None

@dataclass_json
@dataclass
class DatasetRequest(BaseModel):
    """Dataset response ORM model."""
    id: Optional[str] = None
    name: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DatasetResponse(BaseModel):
    """Dataset response ORM model."""
    id: Optional[str]
    name: Optional[str]
    content: Optional[Any]
    files: Optional[List[str]]
    metadata: Optional[Dict[str, Any]] = None


class DatasetsResponse(BaseModel):
    """Dataset response ORM model."""
    items: List[DatasetResponse]


# class Dataset(BaseModel):
#     """Dataset response ORM model."""
#     id: str
#     content: Optional[str] = None
#     metadata: Optional[Dict[str, Any]] = None

class ColumnMapModel(BaseModel):
    question: str = None
    reference: str = None
    additional_params: List[str] = None