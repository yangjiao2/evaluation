# flake8: noqa

class MetricsEvaluationPrompt:
    output_format = """judge-nvbot"""
    eval_template = """[INST] Please act as an impartial judge and evaluate the quality of the responses provided by an AI assistants to the user question displayed below.
You will be given a reference answer and the assistant's answer.

Your evaluation should consider the following dimensions: 
1. Correctness : Assess the accuracy of the assistant's answer by comparing it to the reference answer.

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

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.
[/INST] 

[INST]
Question: {question}

Correct Answer: {correct_answer}

Assistant Answer: {answer}

Evaluate:
[/INST]

""" + output_format

    general = """
    [Question]
    {question}

    [The Start of Reference Answer]
    {ref_answer_1}
    [The End of Reference Answer]

    [The Start of Assistant's Answer]
    {answer}
    [The End of Assistant's Answer]

    Evaluate:
    """

    # output_format_v1 = """{"type":"object","properties":{"Explanation":{"type":"string","description":"A textual explanation or justification for the provided scores."},"Correctness Answer":{"type":"integer","minimum":1,"maximum":5},"Helpfulness":{"type":"integer","minimum":1,"maximum":5},"Empathy":{"type":"integer","minimum":1,"maximum":5},"Conciseness":{"type":"integer","minimum":1,"maximum":5}}}"""
    output_format_v1 = """{'type':'object','properties':{'Explanation':{'type':'string','description':'A textual explanation or justification for the provided scores.'},'Correctness':{'type':'integer','minimum':1,'maximum':5},'Helpfulness':{'type':'integer','minimum':1,'maximum':5},'Empathy':{'type':'integer','minimum':1,'maximum':5},'Conciseness':{'type':'integer','minimum':1,'maximum':5}}}"""

    eval_template_v2 = """Please act as an impartial judge and evaluate the quality of the responses provided by an AI assistants to the user question displayed below.
    You will be given a reference answer and the assistant's answer.

    Your evaluation should consider the following dimensions: 
    1. Correctness : Assess the accuracy of the assistant's answer by comparing it to the reference answer.
    2. Helpfulness : Evaluate the assistant's response based on its helpfulness. The criteria for helpfulness are as follows: {helpfulness_criteria}. Also evaluate the extent to which the response is actionable and can potentially enhance employee productivity.
    If a citation is required in the response, use the citation provided in the assistant's answer for comparison with the Required Citations.
    3. Empathy: Evaluate whether the assistant's answer exhibits empathy, as per the defined empathy criteria: {empathy_expected}
    4. Conciseness: Assess if the assistant's response is {short_answer_expected}, without being overly verbose or omitting essential information.

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
        }},
        "Helpfulness": {{
          "type": "integer",
          "minimum": 1,
          "maximum": 5
        }},
        "Empathy": {{
          "type": "integer",
          "minimum": 1,
          "maximum": 5
        }},
        "Conciseness": {{
          "type": "integer",
          "minimum": 1,
          "maximum": 5
        }}
      }}
    }}

    Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.

    Question: {question}

    Correct Answer: {correct_answer}

    Required Citations: {required_citations}

    Assistant Answer: {answer}

    Evaluate:
    """

    eval_template_required_citations = """<s>[INST] Please act as an impartial judge and evaluate the quality of the responses provided by an AI assistant to the user question displayed below.
You will be given a reference answer and the assistant's answer.

Your evaluation should consider the following dimensions: 
1. Correctness : Assess the accuracy of the assistant's answer by comparing it to the reference answer.
2. Helpfulness : Evaluate the assistant's response based on its helpfulness. Also evaluate the extent to which the response is actionable and can potentially enhance employee productivity.
If a citation is required in the response, use the citation provided in the assistant's answer for comparison with the Required Citations.


Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a 5-point Likert scale on all the dimensions.
Scoring guide:
1 - Strongly Disagree (Very bad) - The answer is completely unrelated to the evaluation criteria defined. 
2 - Disagree (Bad) - The answer may only be partially meeting the evaluation criteria and is not of acceptable quality. 
3 - Neither disagree nor agree (neither bad nor good - okay) - The response is average. It meets some of the evaluation criteria, and misses others.
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
    }},
    "Helpfulness": {{
      "type": "integer",
      "minimum": 1,
      "maximum": 5
    }}
  }}
}}

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.
[/INST]

[INST]
Question: {question}

Correct Answer: {correct_answer}

Required Citations: {required_citations}

Assistant Answer: {answer}

Evaluate:

[/INST]

"""

    eval_template_required_citations_short_answer_expected = """<s>[INST] Please act as an impartial judge and evaluate the quality of the responses provided by an AI assistant to the user question displayed below.
    You will be given a Correct Answer (ground truth) and the Assistant Answer.

    Your evaluation MUST consider the following dimensions: 
    1. Correctness : Assess the accuracy of the assistant's answer by comparing it to the Correct Answer.
    2. Helpfulness : Evaluate the assistant's answer based on its helpfulness. Also evaluate the extent to which the response is actionable and can potentially enhance employee productivity. If a citation is required in the response, check if the Assistant Answer contains the citation or urls provided in the Required Citations.
    3. Conciseness: Assess if the assistant's response's conciseness, scores should be lower if being overly verbose by checking if Short Answer Expected, having long paragraphs, or omitting essential information.


    Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a 5-point Likert scale on all the dimensions.
    Scoring guide:
    1 - Strongly Disagree (Very bad) - The answer is completely unrelated to the evaluation criteria defined. 
    2 - Disagree (Bad) - The answer may only be partially meeting the evaluation criteria and is not of acceptable quality. 
    3 - Neither disagree nor agree (neither bad nor good - okay) - The response is average. It meets some of the evaluation criteria, and misses others.
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
        }},
        "Helpfulness": {{
          "type": "integer",
          "minimum": 1,
          "maximum": 5
        }},
        "Conciseness": {{
          "type": "integer",
          "minimum": 1,
          "maximum": 5
        }}
      }}
    }}

    Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.
    [/INST]

    [INST]
    Question: {question}

    Correct Answer: {correct_answer}

    Required Citations: {required_citations}

    Assistant Answer: {answer}

    Short Answer Expected: {short_answer_expected}
    
    Evaluate:

    [/INST]

    """

    eval_template_perceptor = """"""