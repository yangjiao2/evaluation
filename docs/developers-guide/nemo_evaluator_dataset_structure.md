

##  Dataset Config Schema
To easily test on Evaluation schema, we support loading from local folder and generate evaluation input dataset, so that developers can test on schema.

To load data from local, e.g: from /script folder which will be a child folder.

```json
{
  "DatasetConfig": {
    "Engine": "local",
    "DatasetPath": "script/evaluation-dataset.xlsx"
  }
}

```


## Evaluation Schema

We have 2 types of evaluation, regression evaluation and llm-as-a-judge evaluation. Please check on 
1) doc: https://confluence.nvidia.com/display/ITAppDev/Bot+Evaluation+Onboarding+Guide
2) samples: /asset/
3) script: bot_onboarding.ipynb

## Evaluation dataset structure

questions.jsonl
```json
{"question_id": 0, "category": "general", "turns": ["<QUESTION>"]}

```
judge_prompts.jsonl
```json
{"name": "single-ref-v1", "type": "single", "prompt_template": "<JUDGE_PROMPT>", "category": "general", "output_format": "<STRUCTURE>" }
```
reference_answer 📁
 - reference.jsonl
```json
{"question_id": 0, "answer_id": "random_id_Oqa3l", "model_id": "gpt-4", "choices": [{"index": 0, "turns": ["<REFERENCE_ANSWER>"], "tstamp": 0}
```

judge_prompts_parameters 📁 (all input variables to be formatted into judge prompts, example below:)
 - short_answer.jsonl 
```json
{ question_id: 0, value: "yes" }
```
