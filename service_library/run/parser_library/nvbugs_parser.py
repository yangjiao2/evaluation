import json
import math
from typing import Any

from service_library.nemo_ms.nemo_service_helper import convert_number
from service_library.utils.logging import log_errors

ACCURACY_COLUMN = "Accuracy"
RAGAS_GROUNDNESS_COLUMN = "Context Groundness"
RAGAS_RELEVANCE_COLUMN = "Context Relevance"

def sql_cleanup(request_payload: Any):
    if 'SQL' in request_payload:
        request_payload['SQL'] = request_payload.get("SQL" ,"").replace("\n", " ")
    return request_payload

@log_errors('nvbugs parser for accuracy')
def rescale_accuracy_score(input_dict: Any):
    values = input_dict.get("values")
    row = input_dict.get("row")
    try:
        if row.get(ACCURACY_COLUMN) is not None and not math.isnan(row.get(ACCURACY_COLUMN)):
            values[ACCURACY_COLUMN] = convert_number(row.get(ACCURACY_COLUMN))
        if convert_number(row.get("Status Code")) != 200:
            return {}
        for key in values.keys():
            contains_non_null = key in row and row.get(key) is not None and not math.isnan(
                row.get(key))

            if contains_non_null:
                values[key] = convert_number(row.get(key))
            else:
                score = convert_number(values.get(key, -1))
                if abs(float(score) - 5) < 0.1:
                    values[key] = 1
                elif abs(float(score) - 4) < 0.1:
                    values[key] = 0.8
                else:
                    values[key] = 0

        if row.get(RAGAS_RELEVANCE_COLUMN) is not None and not math.isnan(row.get(RAGAS_RELEVANCE_COLUMN)):
            values[RAGAS_RELEVANCE_COLUMN] = convert_number(row.get(RAGAS_RELEVANCE_COLUMN))
        if row.get(RAGAS_GROUNDNESS_COLUMN) is not None and not math.isnan(row.get(RAGAS_GROUNDNESS_COLUMN)):
            values[RAGAS_GROUNDNESS_COLUMN] = convert_number(row.get(RAGAS_GROUNDNESS_COLUMN))
        return {
            "row": row,
            "values": values
        }
    except Exception as e:
        print (f"Error converting number from {json.dumps(input_dict)}: {e}")
        return values

