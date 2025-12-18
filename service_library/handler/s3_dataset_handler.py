import logging
from io import BytesIO
import os

import pandas as pd
import boto3

from configs.settings import get_settings
from constants import common
from data_models.dataset_handler import DataContentOutputConfig
from service_library.utils.data_helper import byte_content_convertor, get_file_extension
from service_library.utils.logging import log_errors

log = logging.getLogger(__name__)


def _generate_s3_link(bucket_name, folder_name, file_name=""):
    # Construct the S3 object URL
    s3_object_url = f"https://{bucket_name}.s3.amazonaws.com/{folder_name}/{file_name}"

    return s3_object_url


class S3DatasetHandler:
    def __init__(self, project: str, config: dict = None):
        if config is None:
            config = dict()
        settings = get_settings()
        aws_config = settings.config["aws"]
        self.config = config

        self.bucket_name = "nvbot-evaluation"
        self.project = project
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_config["access_key_id"],
            aws_secret_access_key=aws_config["secret_access_key"],
        )

    def create_folder(self, bucket_name, folder_name):
        try:
            # Check if folder name ends with '/'
            if not folder_name.endswith('/'):
                folder_name += '/'

            # Create an empty object (which simulates a folder) in S3
            self.s3.put_object(Bucket=bucket_name, Key=folder_name)

            print(f"Folder '{folder_name}' created successfully in bucket '{bucket_name}'.")
        except Exception as e:
            print(f"Error creating folder: {e}")

    @log_errors('Upload file to S3')
    def upload_file(self, input_file_path, s3_file_path):
        """
        file_path: local file path
        object_key: desired key (file path) in S3
        """
        try:
            print("s3_file_path:", s3_file_path)
            # self.s3.put_object(Body=file_content, Bucket=self.bucket_name, Key=object_key)
            # with open(input_file, "rb") as file:
            #     self.s3.upload_fileobj(file, self.bucket_name, file_path)
            self.s3.upload_file(input_file_path, self.bucket_name, s3_file_path)

            logging.info(f"uploaded file success - '{input_file_path}'")

            return {'status': 'success', 'upload_filepath': input_file_path}
        except Exception as ex:
            return {'status': 'error', 'upload_failure': str(ex)}


    @log_errors('Upload file to S3')
    def upload_file_content(self, input_file_content, s3_file_path):
        """
        file_path: local file path
        object_key: desired key (file path) in S3
        """
        try:
            print("s3_file_path:", s3_file_path)
            # self.s3.put_object(Body=file_content, Bucket=self.bucket_name, Key=object_key)
            # with open(input_file, "rb") as file:
            #     self.s3.upload_fileobj(file, self.bucket_name, file_path)
            # self.s3.upload_file(input_file_path, self.bucket_name, s3_file_path)

            self.s3.put_object(Bucket=self.bucket_name, Key=s3_file_path, Body=input_file_content)
            logging.info(f"uploaded file success - '{s3_file_path}'")

            return {'status': 'success', 'upload_filepath': s3_file_path}
        except Exception as ex:
            return {'status': 'error', 'upload_failure': str(ex)}

    def list_files(self, directory, suffix='', page_size: int = 20, page: int = 1):
        # corner case
        if not directory.endswith('/'):
            directory += '/'
        if not directory.startswith('/'):
            directory = '/' + directory

        logging.info(f"s3 bucket='{self.bucket_name}', prefix='{self.project}_evaluation{directory}'");
        # self.s3.list_objects_v2(Bucket='llm-evaluation', Prefix='avc_evaluation/dataset', Delimiter="/")
        # List objects in the specified directory with the given prefix and suffix
        response = self.s3.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=f'{self.project}_evaluation{directory}',
            Delimiter="/"
        )

        files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith(suffix):
                    files.append({
                        'fileName': key.split('/')[-1],
                        'key': key,
                        'lastModified': obj['LastModified'],
                    })  # Extract file name from the full key
        return files

    @log_errors('Download file from S3')
    def download_file(self, folder_name: str, file_name: str, output_config: DataContentOutputConfig = None):
        try:
            full_filepath = os.path.join(folder_name, file_name)
            print("Download from s3 with full_filepath", full_filepath)
            logging.info(f"Download from s3 with full_filepath {full_filepath}")
            response = self.s3.get_object(Bucket=self.bucket_name, Key=full_filepath)
            file_content = response["Body"].read()
            logging.info(f"downloaded file success - '{full_filepath}'")
            # print("File contents:", file_content)
            # Save the object data to a local file
            if output_config:
                output_format = output_config.Format.lower()
                # print("output_config format", output_format)
                if output_format == 'local':
                    local_file_path = os.path.join(output_config.FilePath or "script", 'temp')
                    with open(local_file_path, 'wb') as f:
                        f.write(file_content)

                if output_format in ['pd', 'dict']:
                    content_format = get_file_extension(file_name)["ext"]

                    df = byte_content_convertor(content_format, output_format, file_content)
                    df = df.fillna("")
                    return df
                    # if output_format == 'pd':
                    #     print(df)
                    #     return df
                    # print("return as dict")
                    # return df.to_dict(orient='records')
                else:
                    return file_content

            return None

        except Exception as e:
            return {'status': 'error', 'result': str(e)}

    def read_file_as_pd(self, file_path):
        try:
            dataset_path = 'dataset/'
            # full_filepath = dataset_path + file
            # bucket = "llm-evaluation"
            # full_filepath = f"{self.project}/{file_name}" # dataset_path + dataset_file
            # response = self.s3.get_object(Bucket=self.bucket_name, Key=full_filepath)
            # response = self.s3.get_object(Bucket="llm-evaluation", Key="evaluation_service/avc_mixtral/dataset/default_regression.xlsx")

            response = self.s3.get_object(Bucket="nvbot-evaluation",
                                          Key="avc_mixtral/dataset/default_evaluation.xlsx")
            file_content = response["Body"].read()
            if file_content is not None:
                # df = read_excel_content_as_df(file_content)
                return pd.read_excel(BytesIO(file_content))
            else:
                return None
        except Exception as err:
            print(f"Error while reading dataset file from S3: {err}")
            return None

# # Example usage:
# bucket_name = "your_bucket_name"
# project_name = "your_project_name"
# s3_handler = AWSS3Handler(bucket_name, project_name)
#
# # Upload a file
# upload_result = s3_handler.upload_file("example_file.txt", "File content here")
# print("Upload result:", upload_result)
#
# # Download a file
# download_result = s3_handler.download_file("example_file.txt")
# print("Download result:", download_result)
#
# # Delete a file
# delete_result = s3_handler.delete_file("example_file.txt")
# print("Delete result:", delete_result)
