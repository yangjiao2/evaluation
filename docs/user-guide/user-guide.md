# IT Enterprise Evaluation Microservice Guides

## Table of Contents
- [IT Enterprise Evaluation Microservice Guides](#it-enterprise-evaluation-microservice-guides)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Use Cases](#use-cases)
    - [1. LLM as Judge](#1-llm-as-judge)
      - [Creating Datasets (LLM as Judge)](#creating-datasets-llm-as-judge)
    - [2. Custom Evaluation](#2-custom-evaluation)


## Overview
Evaluation Microservice is a service that allows IT Enetperise users to evaluate the performance of a our internal bots on a curated dataset. User interact with the platform mainly through UI (TODO). There are several type of evaluations that are currently supported (as of June 10), namely: LLM as Judge and Custom Evaluation.


## Use Cases

### 1. LLM as Judge
Ref: https://huggingface.co/learn/cookbook/en/llm_judge


#### Creating Datasets (LLM as Judge)
**Pre-requisite**:
1. Judge prompts: the prompt that the judge will use to evaluate the performance of the bots. The prompt must be in a jsonl format. At minimum, the judge prompt string must have `{question}`, `{answer}` and `{correct_answer}` for a simple example, see [judge_prompts_simple.jsonl](user-guide/template/judge_prompts_simple.jsonl). NVBot's nvhelp judge prompt uses [judge_prompts.jsonl](user-guide/template/judge_prompts.jsonl), which has more complex structure (see 2).
**Note**: `{answer}` is the completion response from the bot to be evaluated, and must not be provided in the spreadsheet (see 2).

2. User must have prepared a spreadsheet of questions and answers in a spreadsheet file format (xlsx). At minimum, the spreadsheet must have the columns: `ID, question` (if prompt is simple).
Optionally, if judge prompt requires additional variable (e.g. `empathy`), user must add the column to the spreadsheet with exact same header name. For example, the nvbot's judge_prompts.jsonl requires `empathy, helpfulness, short_ans, correct_answer, required_citations` columns, so the spreadsheet must have these columns. See [nvbot_evaluation_small.xlsx](user-guide/template/nvbot_evaluation_small.xlsx) for an example.


**Steps**:
1. Prepare the `.xlsx` containing the questions and answers and optionally, additionally parameters to be passed to templated judge prompt
2. Make a copy of either `create_dataset_judge_*.ipynb` to create contents of dataset to `dataset_output` folder
3. Run the steps in `uploading_judge_datasets.ipynb` to upload the dataset to the evaluation microservice to obtain `Dataset ID` to be used in further judge evaluation

### 2. Custom Evaluation
1. Data should be prepared ahead of time, ideally with `Query`, `Correct Answer`. We also support column mapping which will take original column name and map it in our desired format.

2. We will generate output in both `input.jsonl` and `output.jsonl`, where input will have all info relates to user provided question, and output will have query answers with corresponding to each question. See [input.jsonl](user-guide/template/input.jsonl) and [output.jsonl](user-guide/template/output.jsonl) for an example.


**Steps**:
1. Prepare the `.xlsx` containing the questions and answers.
3. Run the steps in `uploading_custom_datasets.ipynb` to upload the dataset to the evaluation microservice to obtain `Dataset ID` and `path` to be used in further evaluation.
