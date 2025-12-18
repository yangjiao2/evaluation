import os
import pandas as pd
from tqdm import tqdm
import json
import copy
import random
import string
from datetime import datetime


QUESTION_HEADER = "question"
QUESTION_TEMPLATE = {"question_id": 1, "category": "general", "turns": []}

def create_folder_if_not_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def jsonl_writer(data_lists, output_file, mode='w', encoding='utf'):
    with open(output_file, mode) as f:
        for item in data_lists:
            f.write(json.dumps(item) + '\n')

def generate_random_string(length=5):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

    
def build_questions(df: pd.DataFrame):
    questions = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Create a new dictionary for each turn
        question_id = row['ID']
        try:
            if QUESTION_HEADER not in row:
                error_message = f"Question header {QUESTION_HEADER} not exist, row index: {idx}"
                logging.error(error_message)
                print(error_message)
            entry = copy.deepcopy(QUESTION_TEMPLATE)
            entry['turns'].append(row[QUESTION_HEADER])
            entry['question_id'] = int(question_id)
            questions.append(entry)
        except Exception as ex:
            print(f"Row: {row}, exception: {ex}")
    print("Total questions loaded: ", len(questions))
    return questions

def write_questions(output_dir: str = "dataset_output", questions: list = []):
    question_filename = "question.jsonl"
    create_folder_if_not_exists(output_dir)
    fp = os.path.join(output_dir, question_filename)
    jsonl_writer(questions, fp)
    # check file exists
    if os.path.exists(fp):
        print(f"Questions file written to: {fp}")
    else:
        print(f"[ERROR[ Questions file not written to: {fp}")

def build_references(df: pd.DataFrame, reference_header: str = "correct_answer"):
    reference_template = {"question_id": 1, "answer_id": "random_id_" + generate_random_string(5),
                        "model_id": "gpt-4", "choices": [{"index": 0, "turns": []}], "tstamp": 1686286924.844282}

    references = []
    for re_index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Create a new dictionary for each turn
        question_id = row['ID']
        try:
            entry = copy.deepcopy(reference_template)
            turns = entry['choices'][0]['turns']

            if reference_header not in row:
                error_message = f"ERROR: Reference header {reference_header} not exist, row index: {re_index}"
                print(error_message)

            turns.append(row[reference_header])
            choices0 = entry['choices'][0]
            choices0['turns'] = turns
            entry['choices'][0] = choices0
            entry['question_id'] = int(question_id)
            entry['tstamp'] = datetime.timestamp(datetime.now())

            references.append(entry)

        except Exception as ex:
            print(f"Row: {row}, exception: {ex}")

    return references

def write_references(output_dir, references: list = []):
    fn = "reference.jsonl"
    ref_dir = create_folder_if_not_exists(output_dir + "/reference_answer")
    jsonl_writer(references, os.path.join(ref_dir, fn))

def validate_judge_prompts(output_dir, default_prompt_fn: str, df: pd.DataFrame):
    # check if file exists if not create one from default in "/Users/tsalim/Downloads/temp-dl/datasets/defaults/judge_prompts_5_params.jsonl"
    prompt_fn = f"{output_dir}/judge_prompts.jsonl"
    print(prompt_fn)
    if not os.path.exists(prompt_fn):
        os.system(f"cp {default_prompt_fn} {prompt_fn}")
        
    with open(prompt_fn, "r") as f:
        prompts = [json.loads(line) for line in f]

    # find all occurencet of "{\w*}" in the prompt
    import re
    prompt = prompts[0]['prompt_template']
    prompt_vars = [v[1:-1] for v in re.findall(r"{\w*}", prompt)]
    print(f"Prompt variables: {prompt_vars}")
    print(f"Columns in the dataset: {list(df.columns)}")
    diffs = set(prompt_vars) ^ set(df.columns)
    assert diffs == {'ID', 'answer'}, f"Prompt variables differences: {diffs}"

def create_judge_prompt_parameters(output_dir, df: pd.DataFrame):
    for v in list(df.columns):
        if v in ['ID', 'question']:
            continue
        filename = f"{v}.jsonl"
        data = []
        for index, row in df.iterrows():
            entry = {"question_id": int(row["ID"]), "value": row[v]}
            data.append(entry)
        jsonl_writer(data, os.path.join(output_dir, filename))



    def validate_custom_dataset(df: pd.DataFrame):
        return ["Query", "Answer", "Reference"] in df.columns
        