import re
from io import BytesIO
import os
import shutil
import json
from typing import Any, Optional, List
import ast

import pandas as pd
from scipy.stats import ttest_rel

from service_library.utils.logging import log_errors
from datetime import datetime

@log_errors('Flatten dict by 1 degree')
def flatten_dict_by_1_degree(data):
    output = {}

    for key, value in data.items():
        if isinstance(value, dict):
            output.update(value)

    return output

@log_errors('Snake case key convertor')
def convert_to_snake_case_separated_by_dot_notation(dict_data, level=3, current_level=1):
    result = {}

    for key, value in dict_data.items():
        snake_case_key = key.lower().replace(" ", "_")

        if isinstance(value, dict) and current_level < level:
            for sub_key, sub_value in value.items():
                dotted_key = f"{snake_case_key}.{sub_key.lower().replace(' ', '_')}"

                if isinstance(sub_value, dict) and current_level + 1 < level:
                    nested_result = convert_to_snake_case_separated_by_dot_notation(sub_value, level, current_level + 1)
                    for nested_key, nested_value in nested_result.items():
                        result[f"{dotted_key}.{nested_key}"] = nested_value
                else:
                    result[dotted_key] = sub_value
        else:
            result[snake_case_key] = value

    return result


@log_errors('Byte content convertor')
def byte_content_convertor(content_format, output_format, file_content):
    try:
        if output_format.lower() == 'pd' and content_format in [".xlsx"]:
            result = pd.read_excel(BytesIO(file_content), sheet_name=0)
            # print ("status", result.get("status"))
            # print ("read results", result)
            # print ("read results type", type(result))
            return result
        elif output_format.lower() == 'pd' and content_format in [".csv"]:
            return pd.read_csv(BytesIO(file_content))
        print(f"not supported format: {content_format}")
    except Exception as err:
        print(f"Error while reading file from byte io: {err}")
        return None


@log_errors('File content convertor')
def file_content_convertor(output_format, file, parse_dates=False, sheet_name=None):
    content_format = get_file_extension(file)["ext"]
    if output_format.lower() == 'pd':
        if content_format in [".xlsx"]:
            if sheet_name:
                return pd.read_excel(file, parse_dates=parse_dates, sheet_name=sheet_name)
            return pd.read_excel(file, parse_dates=parse_dates)
        elif content_format in [".csv"]:
            return pd.read_csv(file, encoding='utf-8')
    return None


def get_file_extension(file_path):
    root, extension = os.path.splitext(file_path)
    return {
        "root": root,
        "ext": extension
    }


@log_errors("JSONL writer")
def jsonl_writer(data_lists, output_file, mode='w', encoding='utf'):
    with open(output_file, mode) as f:
        for item in data_lists:
            f.write(json.dumps(item) + '\n')
        print("Complete write to file", output_file)


@log_errors("JSON writer")
def json_writer(data_lists, output_file, mode='w', encoding='utf'):
    with open(output_file, mode) as f:
        json.dump(data_lists, f, indent=4)
    print("Complete write to file", output_file)

@log_errors("Update dict")
def update_dict(template_dict, new_values_dict, overwrite = False):
    """
    Recursively update the values of template_dict with the values from new_values_dict.
    This function updates values only if the key exists in template_dict.
    Assumes both dictionaries have a potentially matching structure with nested dictionaries.

    :param template_dict: The dictionary to be updated (only if key exists).
    :param new_values_dict: The dictionary with new values to potentially update.
    """
    result_dict = template_dict.copy()
    for key, value in new_values_dict.items():
        if overwrite or key in result_dict:
            # If the value is a dictionary and the key exists in both, recurse
            if isinstance(value, dict) and isinstance(result_dict[key], dict):
                new_value = update_dict(result_dict[key], value, overwrite)
                result_dict[key] = new_value
            # If the value is not a dictionary, update the value at the key
            elif not isinstance(value, dict):
                result_dict[key] = value
        else:
            result_dict.update({key: new_values_dict[key]})
    return result_dict

@log_errors('Safe Join dict')
def join_dict(dict1, dict2):
    """
    Joins two dictionaries recursively.
    - If values are dictionaries, it applies _join_dict recursively.
    - If values are lists, it concatenates them, ensuring unique elements.
    - For other types, values from dict2 overwrite dict1.
    """
    result = dict(dict1)  # Create a copy of dict1 to avoid modifying the original

    for key, value in dict2.items():
        # If key is in both dictionaries and both values are dictionaries, apply _join_dict recursively
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            result[key] = join_dict(dict1[key], value)
        # If key is in both dictionaries and both values are lists, concatenate unique elements
        elif key in dict1 and isinstance(dict1[key], list) and isinstance(value, list):
            # Since we want values from dict2 to take precedence, we concatenate dict2 first
            result[key] = _unique_list_concat(value, dict1[key])
        # Otherwise, simply overwrite the value from dict2
        else:
            result[key] = value

    return result


def _unique_list_concat(list1, list2):
    """
    Concatenates two lists ensuring all elements are unique.
    Handles unhashable types like dictionaries.
    """
    result = list(list1)
    for item in list2:
        if item not in result:
            result.append(item)
    return result


## evaluation result alert

# def paired_t_test(group1, group2, alpha=0.05):
#     """
#     Perform a paired t-test on two related groups of data and determine if the p-value is statistically significant.
#
#     Returns:
#     dict: A dictionary with the t-statistic, p-value, and significance result.
#     """
#     # Ensure that the input groups have the same length
#     if len(group1) != len(group2):
#         raise ValueError("The two groups must have the same number of observations.")
#
#     # Perform the paired t-test
#     t_statistic, p_value = ttest_rel(group1, group2)
#
#     # Determine if the p-value is statistically significant
#     is_significant = p_value <= alpha
#
#     # Return the\ results as a dictionary
#     return {"t_statistic": t_statistic, "p_value": p_value, "is_significant": is_significant}
# # Example usage
# group1 = [20, 21, 22, 23, 24]
# group2 = [19, 21, 20, 22, 23]
#
# results = paired_t_test(group1, group2)
# print(results)

@log_errors('Move file in local directory')
def move_file(src_file, dst_folder, file_rename=None):
    """
    Move a file to the destination folder. Optionally rename the file.

    :param src_file: Source file path as a string
    :param dst_folder: Destination folder path as a string
    :param file_rename: Optional new name for the file as a string
    """
    # Ensure the destination folder exists
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    if not os.path.exists(src_file):
        return
    # Determine the destination file path
    if file_rename:
        dst_file = os.path.join(dst_folder, file_rename)
    else:
        dst_file = os.path.join(dst_folder, os.path.basename(src_file))
    print(f"move file to destination {dst_file}")
    # Move the file, overwriting if necessary
    if os.path.isfile(src_file):
        print(f"move file: {src_file}")
        shutil.copy2(src_file, dst_file)
    # elif os.path.isdir(src_file):
    #     for file in os.listdir(src_file):
    #         if os.path.isfile(src_file):
    #             print(f"move file: {os.path.join(src_file, file)}")
    #             shutil.copy2(os.path.join(src_file, file), dst_file)
    #


# data manipulation helpers

@log_errors('Json decoder')
def json_decoder(obj: Any):
    """
    Custom JSON decoder function
    :param obj: object to be converted
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj.__dict__

@log_errors('Object to string')
def object_to_string(data: Any) -> str:
    """
    Convert an object to string format
    :param data: Data to be converted
    :return: str formatted object
    """
    try:
        json_data = json.dumps(data, default=lambda o: json_decoder(o), indent=4)
    except Exception as ex:
        print('Error when converting object to string %s', ex)
        json_data = json.dumps(data, default=str)
    return json_data

@log_errors('Safe json loads')
def safe_json_loads(json_string, failback = None) -> Optional[dict | str]:
    """
    Safely load a JSON string, returning the original string if it's not valid JSON.
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as ex:
        try:
            return ast.literal_eval(json_string)
        except (ValueError, SyntaxError):
            print(f"func safe_json_loads ValueError, SyntaxError: {ex}, from `{json_string}`")
            return failback
    except Exception as ex:
        print(f"func safe_json_loads Exception: {ex}, from `{json_string}`")
        return failback

@log_errors('Safe ast literal eval loads')
def safe_ast_literal_eval(value, failback = None) -> Optional[dict | str]:
    """
    Safely load a JSON string or list, returning the original string if it's not valid JSON.
    """
    if isinstance(value, (dict, list)):  # If already a dict or list, return as is
        return value
    if isinstance(value, str):
        value = value.strip()
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
    return value


@log_errors('Filter by Column Filter')
def filter_dataframe_by_column_filters(df: pd.DataFrame, filters: Optional[dict] = None) -> pd.DataFrame:
    if filters is None:
        return df  # If no filters are provided, return the original dataframe.

    for column, regex in filters.items():
        if column not in df.columns:
            print(f"Error: Column '{column}' does not exist in the DataFrame.")
            return df  # Return the original DataFrame if a column is missing.

        # Apply the filter using regex match
        df = df[df[column].astype(str).str.contains(regex, regex=True, na=False)]

    return df

@log_errors('Calculate category metrics')
def calculate_category_metrics(category_column: Optional[str], answer_response: List, scorer: List) -> List:
    if answer_response:
        df = pd.DataFrame(answer_response)
        result_rows = []
        if category_column and category_column in list(df.columns):
            agg_df = df[[category_column] + scorer]

            # Convert the filtered DataFrame back to a dictionary of lists
            # result = filtered_df.to_dict(orient='list')
            category_counts = agg_df[category_column].value_counts().to_dict()

            # Check if 'category' exists in the data
            aggregation_functions = {score: ['mean', 'std'] for score in scorer}
            grouped = agg_df.groupby(category_column).agg(aggregation_functions).reset_index()
            grouped = grouped.round(2)

            # Add count column (2nd column)
            grouped.insert(1, "Count", grouped[category_column].map(category_counts))

            # Compute weighted sum for each scorer
            total_count = len(df)
            for score in scorer:
                grouped[(score, 'weighted_sum')] = (
                        grouped[(score, 'mean')] * grouped["Count"] / total_count
                ).round(2)

            # Compute the overall weighted sum row
            overall_row = {(category_column, ''): "Overall", ("Count", '') : total_count}
            for score in scorer:
                overall_row[(score, 'mean')] = ''
                overall_row[(score, 'std')] = ''
                overall_row[(score, 'weighted_sum')] = round(grouped[(score, 'weighted_sum')].sum(),
                    2)  # Sum of weighted sums

            # Append the overall row
            grouped = pd.concat([grouped, pd.DataFrame([overall_row])], ignore_index=True)

            # Convert to dictionary format
            return {' '.join(map(str, key)): value for key, value in grouped.to_dict(orient='list').items()}


            # return pd.DataFrame(result_rows).T.reset_index(drop=True).to_dict(orient='index')
        return None


@log_errors('Calculate trend')
def calculate_trend(
        df: pd.DataFrame,
        scorer: List[str],
        category_column: str,
) -> pd.DataFrame:
    """
    Determines the trend direction for each scorer based on mean values of each position in the trend lists per source.

    Parameters:
    df (pd.DataFrame): DataFrame containing the source column and trend columns.
    scorer (List[str]): List of column names without the "Trend" suffix.
    source_column (str): Name of the source column in the DataFrame.

    Returns:
    pd.DataFrame: DataFrame where each row is a source and columns are trend directions for each scorer.
    """
    trend_results = {}

    # Iterate over each scorer to calculate trend direction
    for score in scorer:
        trend_column = f"{score} Trend"

        # Group by source to assess trend direction within each group
        for source, group in df.groupby(category_column):
            # Collect each trend position separately
            trend_positions = list(zip(*group[trend_column].tolist()))

            # Calculate mean for each position
            means = [sum(position) / len(position) for position in trend_positions]

            # Determine trend direction based on the means of each position
            if means[-1] > 0 and all(mean >= 0 for mean in means):
                trend_direction = f"trending up (+{means[-1]:.2f})"
            elif means[-1] < 0 and all(mean <= 0 for mean in means):
                trend_direction = f"trending down ({means[-1]:.2f})"
            else:
                trend_direction = ""

            # Update the trend results dictionary
            if source not in trend_results:
                trend_results[source] = {}
            trend_results[source][f"{score} Trend"] = trend_direction

    # Convert trend_results to a DataFrame for better readability and usage
    trend_df = pd.DataFrame.from_dict(trend_results, orient='index').reset_index()
    trend_df.rename(columns={'index': category_column}, inplace=True)

    return trend_df


@log_errors("Drop empty columns")
def drop_df_empty_columns(df, columns):
    # If a list of columns is provided, check only those columns
    if columns:
        columns = [col for col in columns if col in df.columns]
        # Filter columns to drop only those that contain all NaN values in the specified list
        cols_to_drop = [col for col in columns if df[col].isna().all()]
        df = df.drop(columns=cols_to_drop)

    return df


def is_increasing(lst):
    if len(lst) == 1:
        return lst[0] > 0
    # if len(lst) <= 2:
        # return all(x < y for x, y in zip(lst, lst[1:]))
    return all(x <= y for x, y in zip(lst, lst[1:])) and any(x < y for x, y in zip(lst, lst[1:]))

def is_decreasing(lst):
    if len(lst) == 1:
        return lst[0] < 0
    # if len(lst) <= 2:
    #     return all(x > y for x, y in zip(lst, lst[1:]))
    return all(x >= y for x, y in zip(lst, lst[1:])) and any(x > y for x, y in zip(lst, lst[1:]))