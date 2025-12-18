import asyncio
import logging
import uuid
from types import NoneType
from typing import Any, Tuple, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi import Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from requests import Request

from data_models.api.dataset import Dataset, DatasetsResponse, DatasetResponse, DatasetRequest
from data_models.dataset_handler import DataContentOutputConfig
from nvbot_utilities.utils.datadog.custom_span_info import get_current_span, set_span_tags
from service_library.handler.database_handler import DatabaseHandler
from service_library.handler.datadog_handler import DatadogHandler
from service_library.handler.s3_dataset_handler import S3DatasetHandler
import asyncio
logger = logging.getLogger("Results router")

router = APIRouter(
    tags=["Results"]
)

# GetDataset is a FastAPI dependency that will perform stored dataset lookup.
# GetDataset = Depends(dataset_handler)

DATETIME_FASTAPI_FIELD = Query(None, description="Expect in datetime format: %Y-%m-%d. Example: 2024-05-01")


@router.get("/evaluation_history_details")
async def get_evaluation_history_details(
        history_id: int = Query(None, description="ID integer"),
        sortby: str | NoneType = Query(None,
                                       description="Expect in format {fieldname}:{[asc|desc]}, for example: id:desc")
):
    evaluation_history_details = DatabaseHandler({'env': 'dev'}).get_evaluation_history_details(
        history_id=history_id
    )
    return JSONResponse(status_code=200, content=jsonable_encoder(evaluation_history_details, by_alias=False))


async def fetch_history_details_metrics(history: dict):
    history_id = history.get('id')
    if history_id is None: return None
    evaluation_history_details = await asyncio.to_thread(
        DatabaseHandler({'env': 'dev'}).get_evaluation_history_details,
        history_id=history_id
    )
    if len(evaluation_history_details) > 0:
        history["metrics"] = evaluation_history_details[0].get("metrics", {})
    return history


@router.get("/evaluations_history")
async def get_evaluation_history(
        is_active: bool = True,
        id: Optional[int] = Query(None, description="ID integer"),
        run_type: Optional[str] = None,
        created_by_from: Optional[str] = DATETIME_FASTAPI_FIELD,
        created_by_to: Optional[str] = Query(None, description="Expect in datetime format: %Y-%m-%d. Example: 2024-05-01"),
        modified_by_from: Optional[str] = Query(None, description="Expect in datetime format: %Y-%m-%d. Example: 2024-05-02"),
        modified_by_to: Optional[str] = Query(None, description="Expect in datetime format: %Y-%m-%d. Example: 2024-05-02"),
        project: Optional[str] = None,
        project_id: Optional[int] = None,
        tag1: Optional[str] = None,
        tag2: Optional[str] = None,
        status: Optional[str] = Query(None, description="Refers to evaluation run status: STARTED, IN_PROCESS, COMPLETED, FAILED"),
        sortby: Optional[str] = Query(None, description="Expect in format {fieldname}:{[asc|desc]}, for example: id:desc"),
        page: Optional[int] = Query(1, description="Page Number"),
        page_size: Optional[int] = Query(default=50, description="Page size"),
        include_metrics: bool = True,
        history_ids: Optional[str] = Query(None, description="List of history IDs, e.g. 8095&8069&8082")
):
    parsed_history_ids = None
    if history_ids:
        try:
            parsed_history_ids = list(map(int, history_ids.split("&")))
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "Invalid history_ids format. Expected comma-separated integers."})

    # build filters
    filters = {
        "id": id,
        "is_active": is_active,
        "run_type": run_type,
        "created_by_from": created_by_from,
        "created_by_to": created_by_to,
        "modified_by_from": modified_by_from,
        "modified_by_to": modified_by_to,
        "project": project,
        "project_id": project_id,
        "tag1": tag1,
        "tag2": tag2,
        "status": status,
        "sortby": sortby
    }

    # e.g: = 8089 & 8072 & 8083 & 8096
    if parsed_history_ids:
        results = [
            DatabaseHandler({'env': 'dev'}).get_evaluation_history_records(
                filters={"id": history_id},
                pagination={}
            )
            for history_id in parsed_history_ids
        ]
        # Flatten the result if each call returns a dict with "items"
        evaluation_history_list = {
            "items": [res["items"][0] for res in results if res.get("items")],
            "total": len(results)
        }
    else:

        evaluation_history_list = DatabaseHandler({'env': 'dev'}).get_evaluation_history_records(
            filters=filters,
            pagination={
                "page": page,
                "size": page_size
            } if page and page_size else {}
        )

    if include_metrics:
        items = evaluation_history_list.get("items", [])
        items_with_details = await asyncio.gather(
            *(fetch_history_details_metrics(history) for history in items)
        )
        evaluation_history_list["items"] = items_with_details

    return JSONResponse(status_code=200, content=jsonable_encoder(evaluation_history_list, by_alias=False))

# @router.get("/evaluation_result/{evaluation_history_id}", response_model=CommonResponse)
# def get_evaluation_history_by_id(evaluation_history_id: int,
#                                  db: Session = Depends(get_db)):
#     result = EvaluationHistoryService(db).get_evaluation_history(evaluation_history_id)
#     return JSONResponse(status_code=200, content=jsonable_encoder(result))
