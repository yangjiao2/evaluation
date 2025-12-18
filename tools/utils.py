import os
import json


def generate_parameters_mapping(directory):
    parameters_mapping = {}
    judge_prompt_parameters_path = os.path.join(directory, 'judge_prompt_parameters')

    if not os.path.isdir(judge_prompt_parameters_path):
        raise FileNotFoundError(f"Directory {judge_prompt_parameters_path} does not exist")

    for filename in os.listdir(judge_prompt_parameters_path):
        if filename.endswith('.jsonl'):
            param_name = os.path.splitext(filename)[0]
            file_path = os.path.join(judge_prompt_parameters_path, filename)
            with open(file_path, 'r') as file:
                parameters_mapping[param_name] = {}
                for line in file:
                    try:
                        entry = json.loads(line)
                        question_id = entry.get("question_id")
                        value = entry.get("value")
                        if question_id is not None and value is not None:
                            parameters_mapping[param_name][question_id] = value
                    except json.JSONDecodeError:
                        raise ValueError(f"Error decoding JSON in file {file_path}")

    return parameters_mapping


# re-use purpose
def judge_eval_parameters_mapping_validation(directory, validations):
    params_mapping = generate_parameters_mapping(directory)
    prompt_template_content = [None]
    success_messages = []
    error_messages = []

    for validation in validations:
        res, note = validation["func"](directory, prompt_template_content, params_mapping)
        print(res, note)
        if res:
            success_messages.append(f"✅ {validation['message']}")
        elif validation["warning"]:
            msg = f"⚠️ {validation['message']}" if not note else f"⚠️ {validation['message']} ({note})"
            error_messages.append(msg)
        else:
            msg = f"❌ {validation['message']}" if not note else f"❌ {validation['message']} ({note})"
            error_messages.append(msg)

    return success_messages, error_messages, params_mapping, prompt_template_content


def ideal_response_parameters_validation(data):
    success_messages = []
    error_messages = []
    # print (data)
    if isinstance(data, list):
        for line_number, entry in enumerate(data, start=1):
            input_dict = entry.get('input')
            if input_dict and not input_dict.get('ideal_response'):
                print (f"Line number: {line_number}")
                print(f"Prompt: {input_dict['prompt']}\nIdeal Response is missing or empty.\n")
                msg = f"❌ Line number: {line_number} (Prompt {input_dict['prompt']}): `ideal_response` is missing or empty {input_dict.get('ideal_response')}."
                error_messages.append(msg)
            else:
                success_messages.append(f"✅ Line number: {line_number}: `ideal_response` is valid ")
    return success_messages, error_messages