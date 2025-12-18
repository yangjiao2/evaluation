import asyncio
import logging
import mimetypes
import os
import shutil
import uuid
import zipfile
from typing import Any

import requests
from fastapi.responses import FileResponse
from fastapi import APIRouter, HTTPException, Path
from fastapi import Depends
from fastapi.encoders import jsonable_encoder
from fastapi.openapi.models import Response
from fastapi.responses import JSONResponse, StreamingResponse
from requests import Request
from configs.settings import get_settings

from data_models.api.dataset import Dataset, DatasetsResponse, DatasetResponse, DatasetRequest
from data_models.dataset_handler import DataContentOutputConfig
from nvbot_utilities.utils import api_handler
from service_library.constants import LOCAL_EVAL_RESULTS_TMP_FOLDER_WITH_END_SLASH, LOCAL_TMP_FOLDER
from service_library.handler.s3_dataset_handler import S3DatasetHandler
from service_library.nemo_ms.nemo_datastore import NeMoDataStore
from service_library.nemo_ms.nemo_service_helper import download_results_to_local_directory, \
    get_dataset_contents, HEADERS
from service_library.utils.run_helpers import create_folder_if_not_exists

logger = logging.getLogger("dataset router")

router = APIRouter(
    tags=["Dataset"]
)


# @router.get(
#     "/datasets",
#     response_model=DatasetsResponse,
#     response_model_exclude_unset=True,
# )
async def list_datasets(
        *,
        project_id: str,
        page_size: int = 10,
        page: int = 1,
) -> DatasetResponse:
    """
    List all datasets.
    """
    # check_pagination_inputs(page, page_size)
    datasets_list = []

    # Get a total count of items pre-pagination
    # total_datasets = await session.exec(select(func.count(Dataset.id)))  # type: ignore
    # total_dataset_count = total_datasets.one()
    # logger.info(f"{total_dataset_count} datasets found satisfying the query")
    #
    # num_pages = get_num_pages(total_dataset_count, page, page_size)
    # statement = select(Dataset).offset((page - 1) * page_size).limit(page_size)
    # result = await session.exec(statement)
    # dataset_list = result.all()
    # page_dataset_count = len(dataset_list)
    # logger.debug(f"{page_dataset_count} datasets on the current page")

    return DatasetsResponse(
        id="",
        datasets=datasets_list,  # type: ignore
    )


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
        dataset_id: str,
) -> DatasetResponse:
    """
        Get metadata for a dataset. \n
        For example: dataset-HaCynxSt6j4BYcreEnLehA
    """
    try:
        ds_url = get_settings().NEMO_DS_URL
        ds = NeMoDataStore(ds_url, {"id": f"{dataset_id}"})
        create_folder_if_not_exists(LOCAL_TMP_FOLDER)
        dataset = ds.get_dataset()

    except Exception as ex:
        raise Exception(f"Failed to load datasets from datastore, error: {ex}")

    return DatasetResponse(
        id=dataset_id,
        name=dataset.get("name"),
        files=[file['path'] for file in dataset.get('files') if not file['path'].startswith(".")]
    )


@router.get("/datasets/download/{dataset_id}", response_model=DatasetResponse)
async def download_dataset(
        dataset_id: str,
) -> FileResponse:
    """
    Get files for a dataset
    """
    try:
        ds_url = get_settings().NEMO_DS_URL
        ds = NeMoDataStore(ds_url, {"id": f"{dataset_id}"})
        create_folder_if_not_exists(LOCAL_TMP_FOLDER)
        dataset = ds.get_dataset()
        dataset_name = dataset.get("name")
        assert dataset_name is not None, f"Failed to get dataset name from id: {dataset_id}"

        local_dir = os.path.join(LOCAL_TMP_FOLDER, dataset_id)
        local_dir_downloaded = download_results_to_local_directory(ds_url, dataset_name, local_dir)

        assert local_dir_downloaded is not None, f"Failed to download by dataset name: {dataset_name}"
        # Create a ZIP archive of the temporary directory
        logging.info(f"Downloaded and create a zip file at: {local_dir}")
        zip_filename = f"{local_dir}.zip"
        with zipfile.ZipFile(zip_filename, "w") as zipf:
            for root, _, files in os.walk(local_dir_downloaded):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, local_dir))

        # Set response headers for file download
        if not os.path.isfile(zip_filename):
            raise HTTPException(status_code=500, detail="Failed to create zip file")
        # Stream the ZIP file to the client
        with open(zip_filename, "rb") as file:
            contents = file.read()

        if local_dir:
            shutil.rmtree(local_dir, ignore_errors=True)

        return FileResponse(
            path=zip_filename,
            media_type='application/zip',
            filename=f"{dataset_id}.zip"
        )

    except Exception as ex:
        raise Exception(f"Failed to download datasets, error: {ex}")

@router.get("/datasets/download/{dataset_id}/file/contents/{filepath}", response_model=DatasetResponse)
async def download_dataset_contents(
        dataset_id: str,
        file_path: str,
) -> FileResponse:
    """
    Get files for a dataset
    """
    try:
        ds_url = get_settings().NEMO_DS_URL
        ds = NeMoDataStore(ds_url, {"id": f"{dataset_id}"})
        # create_folder_if_not_exists(LOCAL_TMP_FOLDER)
        # dataset = ds.get_dataset()
        # dataset_name = dataset.get("name")
        # assert dataset_name is not None, f"Failed to get dataset name from id: {dataset_id}"
        #
        # local_dir = os.path.join(LOCAL_TMP_FOLDER, dataset_id)
        # local_dir_downloaded = download_results_to_local_directory(ds_url, dataset_name, local_dir)
        #
        # assert local_dir_downloaded is not None, f"Failed to download by dataset name: {dataset_name}"
        # # Create a ZIP archive of the temporary directory
        # logging.info(f"Downloaded and create a zip file at: {local_dir}")
        # zip_filename = f"{local_dir}.zip"
        # with zipfile.ZipFile(zip_filename, "w") as zipf:
        #     for root, _, files in os.walk(local_dir_downloaded):
        #         for file in files:
        #             file_path = os.path.join(root, file)
        #             zipf.write(file_path, os.path.relpath(file_path, local_dir))
        #
        # # Set response headers for file download
        # if not os.path.isfile(zip_filename):
        #     raise HTTPException(status_code=500, detail="Failed to create zip file")
        # # Stream the ZIP file to the client
        # with open(zip_filename, "rb") as file:
        #     contents = file.read()
        #

        # headers = {
        #     'accept': 'application/json',
        # }
        # get_dataset_content_endpoint = "{}/v1/datasets/{}/files/contents/{}".format(ds_url, dataset_id, file_path)
        #
        # # url = f'{ds_url}/{self.USERS_URL.format(userid)}'
        # # log.info('Getting details for user %s', userid)
        # print(get_dataset_content_endpoint)
        # logging.info(f"Getting details for dataset {dataset_id} with file {file_path}")
        # response = requests.get(get_dataset_content_endpoint, headers=HEADERS)

        ds = NeMoDataStore(ds_url, {"id": f"{dataset_id}"})
        create_folder_if_not_exists(LOCAL_TMP_FOLDER)
        dataset = ds.get_dataset()
        dataset_name = dataset.get("name")
        assert dataset_name is not None, f"Failed to get dataset name from id: {dataset_id}"

        local_dir = os.path.join(LOCAL_TMP_FOLDER, dataset_id)
        local_dir_downloaded = download_results_to_local_directory(ds_url, dataset_name, local_dir)


        # file_path = f'/local_dir_downloaded/{file_path}'

        mime_type, _ = mimetypes.guess_type(file_path)
        #
        # with open(file_path, 'wb') as file:
        #     file.write(response.content)
        # if local_dir:
        #     shutil.rmtree(local_dir, ignore_errors=True)
        print ("local_dir_downloaded", os.path.join(local_dir_downloaded, file_path))
        return FileResponse(
            path=os.path.join(local_dir_downloaded, file_path),
            media_type=mime_type,
            filename=f"{os.path.basename(file_path)}"
        )

    except Exception as ex:
        raise Exception(f"Failed to download datasets, error: {ex}")

# @router.get("/{project}/files/contents/{filepath}")
# async def get_dataset_file_content(
#         *,
#         project: str,
#         file_name: str,
# ) -> Any:
#     """
#     Directly download the file content from cloud storage.
#     """
#     try:
#         default_format = DataContentOutputConfig(Format='pd')
#         content = S3DatasetHandler(project).download_file(project, file_name, default_format)
#         return JSONResponse(status_code=200, content=jsonable_encoder({'content': content}, by_alias=False))
#     except Exception as ex:
#         logger.info(ex)
