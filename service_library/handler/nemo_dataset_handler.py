import logging
import os

from service_library.nemo_ms.nemo_service_helper import create_dataset, get_dataset_contents

DS_URL = "https://datastore.dev.llm.ngc.nvidia.com"
EVAL_URL = "https://evaluation-ms-staging-nemo-evaluator.dev.llm.ngc.nvidia.com"


log = logging.getLogger('NemoDatasetHandler')

def _get_file_extension(file_path):
    root, extension = os.path.splitext(file_path)
    return {
        "root": root,
        "ext": extension
    }


class NemoDatasetHandler:
    def __init__(self):
        self.nemo_service_endpoints = {
            'datastore': DS_URL,
            'evaluation': EVAL_URL
        }
        self.dataset_id = None
        self.dataset_name = None

    def upload_to_datastore(self, repo_id, local_path):
        try:
            self.dataset_id = create_dataset(repo_id)

            # Specify the local path to download the results to.
            if not repo_id.startswith("nvidia/"):
                repo_id = f"nvidia/{repo_id}"

            path, ext = _get_file_extension(local_path)
            response = None
            if ext:
                # upload single file
                pass
            else:
                response = upload_dir(local_path, repo_full_name=repo_id, token='mock_token')

            get_dataset_contents(self.dataset_id)
            return response

        except Exception as ex:
            log.error(ex)



