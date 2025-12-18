import re
from io import BytesIO
from typing import List

import pandas as pd
import datetime

from pandas import DataFrame

from constants import common

import urllib.parse
import os

from data_models.api.run_maker import RunMakerRequest
from service_library.constants import LOCAL_TMP_FOLDER_WITH_END_SLASH, LOCAL_TMP_FOLDER
from service_library.utils.logging import log_errors
import uuid
import base64
from dateutil.parser import parse

service_filter = "nvbot-*"


def convert_to_snake_case(text):
    # Step 1: Add an underscore before each capital letter if it is preceded by a lowercase letter or a number
    text = re.sub(r'(?<!\s)(?<=\w)([A-Z])', r'_\1', text).lower()
    # Step 2: Replace all whitespace characters with underscores
    text = re.sub(r'\s+', '_', text)
    # Step 3: Remove any leading underscores (if the original string started with a capital letter)
    text = re.sub(r'^_', '', text)
    # Step 4: Remove all non-alphanumeric characters except underscores
    text = re.sub(r'[^a-z0-9_]', '', text)
    return text


def convert_query_to_snake_case(text):
    # Step 1: Add an underscore before each capital letter if it is preceded by a lowercase letter or a number
    text = re.sub(r'(?<!\s)(?<=\w)([A-Z])', r'_\1', text.lower().strip()).lower()
    # Step 2: Replace all whitespace characters with underscores
    text = re.sub(r'\s+', '_', text)
    # Step 3: Remove any leading underscores (if the original string started with a capital letter)
    text = re.sub(r'^_', '', text)
    # Step 4: Remove all non-alphanumeric characters except underscores
    text = re.sub(r'[^a-z0-9_]', '', text)
    return text


def generate_short_uuid(length=None):
    """
    Generates a short UUID by creating a random UUID,
    converting it to bytes, encoding it in Base64,
    and stripping any padding characters.
    """
    allowed_chars_pattern = re.compile(r'[a-zA-Z0-9]')
    # Generate a random UUID and convert the UUID to bytes
    uuid_bytes = uuid.uuid4().bytes
    base64_uuid = base64.urlsafe_b64encode(uuid_bytes)
    # Decode the Base64 bytes to a string and strip padding
    short_uuid = base64_uuid.decode('utf-8').rstrip('=')
    filtered_uuid = ''.join(allowed_chars_pattern.findall(short_uuid))

    if length:
        return filtered_uuid[:length]

    return filtered_uuid


def generate_datadog_trace_url(service_filter, trace_id, env_filter=os.getenv("ENV", "dev")):
    base_url = "https://nvitprod.datadoghq.com/logs"

    if env_filter:
        query = f"service:{service_filter} env:{env_filter} trace_id:\"{trace_id}\""
    else:
        query = f"service:{service_filter} trace_id:\"{trace_id}\""

    # URL encode the query
    encoded_query = urllib.parse.quote(query)

    return f"{base_url}?query={encoded_query}&agg_m=count&agg_m_source=base&agg_t=count&cols=host,service&fromUser=true"


# # Example usage:
# service = "nvbot-*"
# env = "sandbox"
# trace_id = "6643cb2400000000b8a01408aec0ca22"
#
# url = generate_datadog_url(service, env, trace_id)
# print(url)

@log_errors("Generate local file")
def generate_regression_filename(user):
    str_datetime = datetime.datetime.now().strftime("%m_%d-%H%M")
    if user:
        return f"{str_datetime}-{user}.xlsx"
    output_filename = str_datetime + ".xlsx"
    return output_filename


@log_errors("Generate local folder/file by eval_type")
def generate_nemo_eval_local_folder_path(request: RunMakerRequest, eval_type: str):
    str_datetime = datetime.datetime.now().strftime("%m%d-%H%M")
    # if eval_type.lower() == "automatic":
    #     file_path = f"script/nemo_eval/{request.Project}/{eval_type}-{str_datetime}"
    #     print(f"file path: {file_path}")
    #     return file_path
    # else:
    if request.Project or request.ProjectId:
        create_folder_if_not_exists(LOCAL_TMP_FOLDER_WITH_END_SLASH)
        os.chmod(LOCAL_TMP_FOLDER_WITH_END_SLASH, 0o775)

        create_folder_if_not_exists(f"{LOCAL_TMP_FOLDER_WITH_END_SLASH}{request.Project}")
        parent_folder = request.Project if request.Project else f"ProjectId-{request.ProjectId}"
        folder_path = os.path.join(LOCAL_TMP_FOLDER, parent_folder, f"{eval_type}-{str_datetime}")
        create_folder_if_not_exists(folder_path)

        print(f"Generated folder path: {folder_path}")

        return folder_path


@log_errors("Get formatted_datetime with delta")
def get_formatted_datetime(datetime_timestamp=None, delta_dict=None):
    """
    Returns a formatted datetime string in ISO 8601 format.
    Returns:
        str: The formatted datetime string in "YYYY-MM-DDTHH:MM" format.
    """
    if datetime_timestamp is None:
        # Use the current datetime
        datetime_timestamp = datetime.datetime.now()

    # Extract values from the dictionary
    delta_months = delta_dict.get('months', 0)
    delta_days = delta_dict.get('days', 0)
    delta_hours = delta_dict.get('hours', 0)
    delta_minutes = delta_dict.get('minutes', 0)

    if delta_days != 0 or delta_hours != 0 or delta_minutes != 0 or delta_months != 0:
        new_date = datetime_timestamp + datetime.timedelta(days=delta_days + delta_months * 31, hours=delta_hours,
                                                           minutes=delta_minutes)
        return new_date.strftime("%Y-%m-%dT%H:%M")
    return datetime_timestamp.strftime("%Y-%m-%dT%H:%M")


def is_timestamp_in_range(timestamp, start_range=None, end_range=None):
    """
    Checks if a particular timestamp is within the specified start and end range.

    Parameters:
    timestamp (str): The particular timestamp in "%Y-%m-%d %H:%M:%S.%f" format.
    start_range (str): The start of the range in "%Y-%m-%dT%H:%M" format. Can be None.
    end_range (str): The end of the range in "%Y-%m-%dT%H:%M" format. Can be None.

    Returns:
    bool: True if the timestamp is within the range, False otherwise.
    """
    timestamp_dt = parse(timestamp)

    # Convert start_range and end_range to datetime if provided
    start_dt = parse(start_range) if start_range else None
    end_dt = parse(end_range) if end_range else None

    if start_dt and end_dt:
        return start_dt <= timestamp_dt <= end_dt
    elif start_dt:
        return start_dt <= timestamp_dt
    elif end_dt:
        return timestamp_dt <= end_dt
    else:
        return True  # No range provided, always return True


@log_errors("Generate local folder/file by eval_type")
def generate_unique_datastore_path(prefix: str, eval_type: str):
    str_datetime = datetime.datetime.now().strftime("%m%d-%H%M")
    unique_id = generate_short_uuid(3)[:3]

    return f"{prefix}-{eval_type}-{str_datetime}-{unique_id}" if eval_type else f"{prefix}-{str_datetime}-{unique_id}"

from itertools import cycle

def get_shades_of_gray(n):
    """Generate n shades of gray as RGB colors."""
    step = int(255 / (n + 1))  # Ensure contrast between shades
    return [f"#{value:02x}{value:02x}{value:02x}" for value in range(step, 256, step)]



@log_errors("Generate excel output")
def generate_excel_output(file_path: str, data_sheet_name_styles: List[dict | str] = [], dfs: List[DataFrame] = [],
                          sheet_names: List[str] = [],
                          **metrics_collections) -> str:
    blue_color = '#b3d9ff'
    gray_colors = ["#f2f2f2", "#d9d9d9"]  # Very light and a bit darker gray
    row_height = 20
    if not file_path.endswith(".xlsx"):
        file_path += ".xlsx"

    sheet_name = ""

    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        for index, df in enumerate(dfs):
            sheet_name_style = data_sheet_name_styles[index] if len(data_sheet_name_styles) > index else f"Data{index}"
            if isinstance(sheet_name_style, str):
                sheet_name = sheet_name_style
                sheet_style = {
                    "index": [index for index in range(df.shape[0]) if index % 2 == 0],
                    "color": blue_color
                }
            else:
                sheet_name = list(sheet_name_style.keys())[0] or f"Data{index}"
                sheet_style = list(sheet_name_style.values())[0]

            df.to_excel(writer, sheet_name=sheet_name, index=False)
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]

            # add group columns background color
            color_cycle = cycle(gray_colors)

            groups = {}
            for col in df.columns:
                if "." in col:
                    base_name, index_part = col.rsplit(".", 1)
                    if index_part.isdigit():
                        groups.setdefault(base_name, []).append(col)

            # Assign alternating gray colors to groups
            group_colors = {group: next(color_cycle) for group in groups.keys()}

            # Apply group colors
            for group, columns in groups.items():
                color = group_colors[group]
                for col in columns:
                    col_idx = df.columns.get_loc(col)  # Get column index
                    worksheet.set_column(col_idx, col_idx, None, workbook.add_format({'bg_color': color}))

            for idx in range(df.shape[0] + 1):
                if idx in sheet_style.get("index"):
                    worksheet.set_row(idx, row_height,
                                      workbook.add_format({'bg_color': sheet_style.get("colors", blue_color)}))
                # else:
                #     worksheet.set_row(idx, row_height, workbook.add_format({'bg_color': colors[1]}))


        header_format = {'bold': True, 'text_wrap': True, 'valign': 'center', 'fg_color': '#D7E4BC', 'border': 1}
        sheet_name = ""
        for index, metric in enumerate(metrics_collections.values()):
            if not metric:
                continue
            prev_sheet_name = sheet_names[index - 1] or ""

            sheet_name = sheet_names[index] if index < len(sheet_names) else f"Sheet{index}"
            if sheet_name == prev_sheet_name and index != 0:
                worksheet = writer.sheets[sheet_name]
                startrow = worksheet.dim_rowmax + 2

                # NEED REVISIT
                df_1 = pd.DataFrame.from_dict(metric)
                df_1.to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=True)

            else:
                if isinstance(metric, dict):

                    value_is_list = (isinstance(list(metric.values())[0], list))
                    df_2 = pd.DataFrame.from_dict(metric)
                    if value_is_list:
                        df_2.to_excel(writer, sheet_name=sheet_name,
                                      index=False, header=True)

                    else:
                        df_2.to_excel(writer, sheet_name=sheet_name)

    print(f"🍓Result saved to {file_path}")
    return file_path


def create_folder_if_not_exists(folder_path):
    """
    Check if the folder exists at the specified path.
    If not, create the folder.

    Args:
    - folder_path (str): The path of the folder to check/create.
    """
    print("folder_path created:", folder_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        os.chmod(folder_path, 0o775)
        print(f"\nFolder created at: {folder_path}")
        return folder_path
    else:
        print(f"\nFolder already exists at: {folder_path}")
