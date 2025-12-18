import logging
import os
from http.client import HTTPException
from typing import Optional
import huggingface_hub as hh
import requests

from data_models.api.run_maker import StorageType
from service_library.nemo_ms.nemo_service_helper import create_dataset_and_upload_folder, get_dataset_contents, \
    create_dataset
from service_library.utils.logging import log_errors
from service_library.utils.request_helpers import RequestType, make_request
from service_library.utils.run_helpers import create_folder_if_not_exists, generate_short_uuid

DS_URL = "https://datastore.dev.llm.ngc.nvidia.com"


class NeMoDataStore:
    def __init__(self, url: str, config: dict = None):
        self.url = url
        self.name = config.get("name", None) if config else None
        self.upload_dir = config.get("upload_dir", None) if config else None

        self.id = config.get("id", None)
        self.req_result = None
        self.datastore_content = {
            "Name": self.name,
            "Engine": StorageType.DATASTORE.value,
            "DatasetId": self.id,
            "DatasetFolder": None,
            "Files": []
        }

    def create_dataset_only(self, name: str, description: str) -> str:
        params = {
            "name": name,
            "description": description,
        }
        self.req_result = make_request(
            RequestType.POST, self.url, "datasets", json=params,
        )
        return self.req_result['id']


    def get_dataset(self):
        return get_dataset_contents(self.url, self.id)

    @log_errors('Create Dataset and upload to NeMo Datastore')
    def create_via_hfapi(self, path_in_repo: str = ".") -> dict:
        # assume path is folder (directory)
        repo_name = self.name
        path = self.upload_dir # os.path.join(self.upload_dir, path_in_repo)

        # TODO: validate dataset id is unique...
        print("Datastore upload dir from: ", path)
        [self.id, remote_dir] = create_dataset_and_upload_folder(self.url, repo_name, path, path_in_repo)

        files = get_dataset_contents(self.url, self.id, file_path_only=True)
        # files = [file[len(path_in_repo):] for file in files]
        print ("Files: ", files)
        processing_message = f"Datastore uploaded files: {files}, to Datastore Id: {self.id}"
        print(processing_message)
        logging.info(processing_message)

        # def get_common_prefix_and_trim(paths):
        #     # Find the common prefix
        #     common_prefix = os.path.commonprefix(paths)
        #
        #     # Ensure the common prefix ends with '/'
        #     if not common_prefix.endswith('/'):
        #         common_prefix = common_prefix.rsplit('/', 1)[0]
        #
        #     # Trim the common prefix from each path
        #     trimmed_paths = [path[len(common_prefix) + 1:] if path.startswith(common_prefix + '/') else path for path in
        #                      paths]
        #
        #     return common_prefix, trimmed_paths
        #
        # common_prefix, trimmed_files = get_common_prefix_and_trim(files)

        self.datastore_content = {
            "Engine": StorageType.DATASTORE.value,
            "DatasetId": self.id,
            "DatasetFolder":  remote_dir, #os.path.join(remote_dir, path_in_repo),
            "Name": self.name,
            "Files": files
        }
        logging.info(f"Dataset {self.id} created at {remote_dir}, contains {files}.")

        return {"id": self.id, "metadata": self.datastore_content}

    def delete(self, dataset_id: str):
        self.req_result = make_request(
            RequestType.DELETE, self.url, f"datasets/{dataset_id}",
        )

    # def upload_data(self, dataset_id: str, ds_dir_name: str, file_to_upload: str) -> None:
    #     headers = {
    #         "Accept": "application/json",
    #     }
    #
    #     with open(file_to_upload, 'rb') as f:
    #         files = {'file': f}
    #         filename = os.path.basename(file_to_upload)
    #         model_endpoint = f"datasets/{dataset_id}/files/contents/{ds_dir_name}/{filename}"
    #         print(f"Uploading '{filename}' to '{model_endpoint}'...")
    #
    #     self.req_result = make_request(
    #         RequestType.POST,
    #         self.url,
    #         model_endpoint,
    #         headers=headers,
    #         files=files,
    #     )

    @log_errors('Upload file to Datastore')
    def upload_file(self, file_path, dest_file_name='answers.xlsx'):
        headers = {
            'accept': 'application/json',
        }
        files = {
            'file': file_path,
        }

        # pattern
        assert self.datastore_content["id"] is not None, "Datastore id is needed for upload file"
        url = f"{self.url}/v1/datasets/{self.datastore_content['id']}/files/contents/{dest_file_name}"
        response = requests.post(url, headers=headers, files=files)
        return response.json()

    # @log_errors('Upload result to Datastore')
    # def create_dataset_and_upload_result_file(self, dataset_name: str, file_path: str,
    #                                           dest_file_name: str = 'answers.xlsx'):
    #     headers = {
    #         'Accept': 'application/json',
    #         'Content-Type': 'application/json'
    #     }
    #     files = {
    #         "file": file_path,
    #         "message": ""
    #     }
    #
    #     # create evaluation results
    #     evaluation_results_url = f"{self.url}/v1/evaluation_results"
    #     suffix = generate_short_uuid()[:5]
    #     dataset_name += f"-{suffix}"
    #     # evaluation_results_creation_response = requests.post(
    #     #     url=evaluation_results_url,
    #     #     headers=headers,
    #     #     json={
    #     #         "name": dataset_name,
    #     #         "description": "results"
    #     #     })
    #     dataset_id = create_dataset(
    #         self.url,
    #         dataset_name,
    #         "results"
    #     )
    #     if dataset_id is None:
    #         print("dataset exist for repo_name")
    #     logging.info('created dataset id:', dataset_id)
    #
    #
    #     # upload to eval_result created
    #     evaluation_results_upload_url = f"{self.url}/v1/evaluation_results/{eval_result_id}/files/contents/{dest_file_name}"
    #     headers = {
    #         "accept": "application/json",
    #         "Content-Type": "multipart/form-data"
    #     }
    #     files = {
    #         "file": (f"{file_path}", open(f"{file_path}", "rb"), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    #     }
    #     data = {
    #         "message": "this is a result file"
    #     }
    #     response = requests.post(evaluation_results_upload_url, headers=headers, files=files, data=data)
    #     print("upload response", response)
    #     return eval_result_id, response.json()

    def get_attributes(self):
        return vars(self)

    @log_errors("Download results from datastore")
    def download_results_to_local_directory(self, repo_id, local_directory="eval-results"):
        """ repo id can be either dataset or eval"""
        # Specify the local path to download the results to.
        # if not repo_id.startswith("nvidia/"):
        #     repo_id = f"nvidia/{repo_id}"
        # print("result download to local dir: ", local_directory)
        path = os.path.join(local_directory, repo_id)
        create_folder_if_not_exists(path)
        download_directory = path or local_directory

        datastore_directory = f"nvidia/{repo_id}"
        logging.info(f"Downloaded from datastore directory: {datastore_directory}")
        api = hh.HfApi(endpoint=self.url, token="token_mock")
        # Download the results into the current directory.
        api.snapshot_download(
            repo_id=datastore_directory,
            repo_type='dataset',
            cache_dir=None,
            local_dir=download_directory,
            local_dir_use_symlinks=False,
        )
        return download_directory
