import os
import json
import re
import zipfile

import pandas as pd
import streamlit as st
import typing
from utils import generate_parameters_mapping, judge_eval_parameters_mapping_validation, ideal_response_parameters_validation


def find_params_in_prompt(text: str) -> list:
    # Regular expression pattern to match words inside curly brackets
    pattern = r'\{(\w+)\}'
    matches = re.findall(pattern, text)
    return matches

def highlight_placeholders(text, invalid_params=[]):
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("{{", "__DOUBLE_LEFT_BRACE__").replace("}}", "__DOUBLE_RIGHT_BRACE__")
    pattern = r'(\{(\w+)\})'
    def replace_with_color(match):
        param = match.group(2)
        if param in invalid_params:
            return f'<span style="background-color: lightcoral;">{match.group(1)}</span>'
        else:
            return f'<span style="background-color: yellow;">{match.group(1)}</span>'

    highlighted_text = re.sub(pattern, replace_with_color, text)
    highlighted_text = highlighted_text.replace("__DOUBLE_LEFT_BRACE__", "{{").replace("__DOUBLE_RIGHT_BRACE__", "}}")
    return highlighted_text

def validate_judge_prompts_jsonl(directory, prompt_template_content, params):
    judge_prompts_path = os.path.join(directory, 'judge_prompts.jsonl')
    if os.path.isfile(judge_prompts_path):
        with open(judge_prompts_path, 'r') as file:
            try:
                content = json.load(file)
                prompt_template_content[0] = content.get('prompt_template', None)
                return True, None
            except json.JSONDecodeError:
                st.error("Error decoding JSON from judge_prompts.jsonl")
                return False, "Error decoding JSON from judge_prompts.jsonl"
    else:
        return False, None

def validate_judge_prompt_parameters(directory, prompt_template_content, params):
    judge_prompt_parameters_path = os.path.join(directory, 'judge_prompt_parameters')
    return os.path.isdir(judge_prompt_parameters_path), None

def validate_judge_prompt_has_answer(directory, prompt_template_content, params):
    placeholders = find_params_in_prompt(prompt_template_content[0])
    if 'answer' not in placeholders:
        return False, "Ensure that {answer} is used in the judge prompt"
    return True, None

def validate_judge_prompt_has_questions(directory, prompt_template_content, params):
    placeholders = find_params_in_prompt(prompt_template_content[0])
    if 'question' not in placeholders:
        return False, "Required: {question} placeholder is missing in prompt_template ("
    return True, None

def validate_judge_prompt_has_undefined_params(directory, prompt_template_content, params):
    params_in_prompt = find_params_in_prompt(prompt_template_content[0])
    provided_param_keys = params.keys()
    provided_param_keys = [item for sublist in provided_param_keys for item in sublist]
    provided_param_keys.extend(["question", "answer"])
    params_difference = set(params_in_prompt) ^ set(provided_param_keys)
    print(params_difference)
    if len(params_difference) != 0:
        return False, f"Params in prompt template: {params_difference}. Ensure that {params_difference}.json file exists !"
    return True, None


def validate_params_question_ids(directory, prompt_template_content, params):
    # params: { A: {1: "asdas", 2: "asda"}, B: {1: "asdas", 2: "asda"} }
    questions_file = os.path.join(directory, 'question.jsonl')
    try:
        with open(questions_file, 'r') as file:
            questions_data = [json.loads(line) for line in file]
            question_ids = [entry['question_id'] for entry in questions_data]
    except FileNotFoundError:
        return False, "question.jsonl file not found in the main directory"

    mismatch = []
    for param,questions in params.items():
        x = list(questions.keys())
        if set(x) != set(question_ids):
            mismatch.append((param, set(x) ^ set(question_ids)))
    if mismatch:
        return False, f"Question IDs mismatch in params files: {mismatch}"

    return True, None

def add_validation(message, validation_func, warning=False):
    return {"message": message, "func": validation_func, "warning": warning}

def extract_zip(uploaded_zip):
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall('extracted_folder')
    return 'extracted_folder'

def main():
    st.title("LLM Judge Validator")
    # Directory picker
    directory = st.text_input("Select a directory to validate", value="../docs/user-guide/dataset_output")

    # List of validations
    validations: list[dict[str, typing.Any]] = [
        add_validation("File: judge_prompts.jsonl", validate_judge_prompts_jsonl),
        add_validation("Folder: judge_prompt_parameters", validate_judge_prompt_parameters),
        add_validation("No parameters mismatch", validate_params_question_ids, warning=False),
        add_validation("prompt_template has {answer} placeholder", validate_judge_prompt_has_answer),
        add_validation("prompt_template has {question} placeholder", validate_judge_prompt_has_questions),
    ]

    if st.button("Validate Judge Data"):
        if directory:
            [success_messages, error_messages, params_mapping, prompt_template_content] = judge_eval_parameters_mapping_validation(directory, validations)
            st.write("Validation Results:")

            if success_messages:
                for msg in success_messages:
                    st.markdown(f"<p style='color: green;'>{msg}</p>", unsafe_allow_html=True)

            if error_messages:
                for msg in error_messages:
                    st.markdown(f"<p style='color: red;'>{msg}</p>", unsafe_allow_html=True)

            # Display placeholders found in .prompt_template
            placeholders = []
            if prompt_template_content[0]:
                placeholders = find_params_in_prompt(prompt_template_content[0])
                if placeholders:
                    st.write("Placeholders found in .prompt_template:")
                    st.write(placeholders)

            # Display content of .prompt_template at the end
            if prompt_template_content[0]:
                valid_extra_params = params_mapping.keys()
                invalid_params = set(placeholders) - set(valid_extra_params) - set(["question", "answer"])
                print(placeholders)
                print(valid_extra_params)
                print(invalid_params)

                highlighted_content = highlight_placeholders(prompt_template_content[0], invalid_params)
                st.markdown("<h3 style='color: green;'>Judge Prompt Template</h3>", unsafe_allow_html=True)
                # st.write(prompt_template_content[0])
                st.markdown(f"<div style='border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9;'>{highlighted_content}</div>", unsafe_allow_html=True)

        else:
            st.error("Please select a directory")

    st.title("Custom Dataset Validator")
    uploaded_file = st.file_uploader("Upload a CSV/excel file")
    custom_file_content = st.text_area("or input json file (input.jsonl, output.jsonl) below")

    error_messages2 = []
    if st.button("Validate Custom Data"):
        print ("Validate Custom Data...")
        [custom_success_message, custom_error_messages] = ideal_response_parameters_validation(json.loads(custom_file_content))

        if custom_error_messages:
            for msg in custom_error_messages:
                st.markdown(f"<p style='color: red;'>{msg}</p>", unsafe_allow_html=True)

        if len(custom_success_message) == len(json.loads(custom_file_content)):
            st.markdown(f"<p style='color: green;'>Valid! </p>", unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File successfully loaded")

            # Example validation: Check for missing values
            if df.isnull().values.any():
                st.error("The file contains missing values")
            else:
                st.success("No missing values found")

        except Exception as e:
            st.error(f"Error loading file: {e}")


if __name__ == "__main__":
    main()
