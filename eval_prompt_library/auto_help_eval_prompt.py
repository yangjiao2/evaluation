# flake8: noqa


class AutoHelpEvaluationPrompt:
    output_format = """judge-nvbot"""

    eval_template_auto_help = """[INST] Please act as an impartial judge and evaluate the quality of the responses provided by an AI assistants to the user question displayed below.
You will be given a reference correct answer and the assistant's answer.

Your evaluation should consider the following dimensions: 
1. Correctness: Assess the accuracy of the assistant's answer by comparing it to the correct answer. The steps provided in the correct answer should be present in the assistant's answer. Do the same check for the citations provided in the correct answer.

Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a 5-point Likert scale on all the dimensions.
Scoring guide:
1 - Strongly Disagree (Very bad) - The answer is completely unrelated to the evaluation criteria defined. 
2 - Disagree (Bad) - The answer may only be partially meeting the evaluation criteria and is not of acceptable quality. 
3 - Neither disagree nor agree (neither bad nor good - okay) - The response is average. It meets some of the evaluation criterion, and misses others.
4 - Agree (good) - The answer  satisfies the evaluation criteria but could be framed better for this task, or misses the detail required to make it perfect.
5 - Strongly Agree (very good or excellent) - The answer not only satisfies, but exceeds the evaluation criteria.

Strictly follow the JSON schema to output your response.

{{
  "type": "object",
  "properties": {{
    "Explanation": {{
      "type": "string",
      "description": "A textual explanation or justification for the provided scores.",
    }},
    "Correctness Answer": {{
      "type": "integer",
      "minimum": 1,
      "maximum": 5
    }}
  }}
}}

Important Guidelines:
    - Avoid position biases and ensure response order does not influence your judgment.
    - Do not let response length influence your evaluation.
    - Be objective and impartial without favoring specific assistant names.
    - If Assistant Answer is blank score should be a 1.
[/INST]

[INST]
Question: {question}

Correct Answer: {correct_answer}

Assistant Answer: {answer}

Evaluate:
[/INST]"""
