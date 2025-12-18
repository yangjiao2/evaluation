import json
import logging
import os

import pandas as pd

from data_models.api.run_maker import StorageType, DatasetConfig
from data_models.dataset_handler import DataContentOutputConfig
from service_library.constants import LOCAL_TMP_FOLDER
from service_library.handler.database_handler import DatabaseHandler
from service_library.handler.s3_dataset_handler import S3DatasetHandler
from service_library.nemo_ms.nemo_datastore import NeMoDataStore
from service_library.nemo_ms.nemo_service_helper import download_results_to_local_directory
from service_library.utils.data_helper import file_content_convertor, filter_dataframe_by_column_filters
from service_library.utils.logging import log_errors
from configs.settings import get_settings
from service_library.utils.run_helpers import create_folder_if_not_exists

logger = logging.getLogger(__name__)


@log_errors('Load data from datasetconfig')
def load_data_from_datasetconfig(project_name: str, dataset_config: DatasetConfig) -> pd.DataFrame:
    # for data not in datastore
    logging.info(f"Load data from datasetconfig: {project_name} via {json.dumps(dataset_config.dict())}")
    print(f"Load data from datasetconfig: {project_name} via {json.dumps(dataset_config.dict())}")
    df = None
    if dataset_config is not None:

        if dataset_config.Engine.lower() == "database":
            if dataset_config.HistoryId:
                history_datasetconfig = DatabaseHandler({'env': 'dev'}).get_detail_result_as_datasetconfig(
                    history_id=dataset_config.HistoryId)
                if history_datasetconfig:
                    dataset_config = history_datasetconfig

        if dataset_config.Engine.lower() == StorageType.S3.value:
            df = S3DatasetHandler(
                project_name,
                dataset_config.dict()
            ).download_file(
                folder_name=dataset_config.Name,
                file_name=dataset_config.DatasetPath,
                output_config=DataContentOutputConfig(
                    Format='pd'))
            print ("Data derived from s3")
        elif dataset_config.Engine.lower() == StorageType.DATASTORE.value.lower():
            ds_url = get_settings().NEMO_DS_URL
            dataset_id = dataset_config.DatasetId
            print(f"Load data from datastore: id {dataset_config.DatasetId}")
            ds = NeMoDataStore(ds_url, {"id": f"{dataset_id}"})
            create_folder_if_not_exists(LOCAL_TMP_FOLDER)
            dataset = ds.get_dataset()
            dataset_name = dataset.get("name")
            assert dataset_name is not None, f"Failed to get dataset name from id: {dataset_id}"

            local_dir = os.path.join(LOCAL_TMP_FOLDER, dataset_id)
            local_dir_downloaded = download_results_to_local_directory(ds_url, dataset_name, local_dir)

            file_path = dataset_config.Files
            if len(file_path) >= 1:
                file_path = dataset_config.Files[0]
                df = pd.read_excel(os.path.join(local_dir_downloaded, file_path))
            print("Data derived from datastore")
        elif dataset_config.Engine.lower() == "local":
            # HACK: if want to use local file, can use this to read local files
            # df = pd.read_csv('script/avc_eval_dataset.csv', encoding='utf-8')
            # df = pd.read_excel('script/nvhelp_1.xlsx')
            file = dataset_config.RunFile
            df = file_content_convertor("pd", file)
            print(f"Data derived from file: {dataset_config.RunFile}")

        elif dataset_config.Engine.lower() == "dataframe":
            df = pd.DataFrame(dataset_config.Data)

    if df is None:
        return None

    data_limit = dataset_config.DataLimit
    data_shuffle_seed = dataset_config.DataShuffleSeed
    dataset_filters = dataset_config.Filters

    if data_limit is not None and df.shape[0] > data_limit:
        df = df[0:data_limit]
    if data_shuffle_seed is not None:
        df = df.sample(frac=1, random_state=abs(data_shuffle_seed))
    if dataset_filters:
        df = filter_dataframe_by_column_filters(df, dataset_filters)
    df = df.where(pd.notna(df), "")
    return df


@log_errors('Check if Datasetconfig is same')
def is_same_datasetconfig(datasetconfig1: DatasetConfig, datasetconfig2: DatasetConfig) -> bool:
    if datasetconfig1.Engine != datasetconfig2.Engine:
        return False

    if datasetconfig1.Engine.lower() == "s3":
        if datasetconfig1.DatasetFolder != datasetconfig2.DatasetFolder \
                or datasetconfig1.DatasetPath != datasetconfig2.DatasetPath \
                or datasetconfig1.DataLimit != datasetconfig2.DataLimit \
                or datasetconfig1.DataShuffleSeed != datasetconfig2.DataShuffleSeed:
            return False

    elif datasetconfig1.Engine.lower() == "local":
        if datasetconfig1.RunFile != datasetconfig2.RunFile \
                or datasetconfig1.DataLimit != datasetconfig2.DataLimit \
                or datasetconfig1.DataShuffleSeed != datasetconfig2.DataShuffleSeed:
            return False
    return True