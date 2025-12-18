# flake8: noqa

class NVbugsEvaluationPrompt:
    output_format = """judge-nvbot"""

    eval_template = """
{prompts_data}
    
Question: {question}
Response: {answer}

Now evaluate the quality of the response given the data and the question. Evaluate it on a scale of 0-5. 
If the comparison involves comparing two floating point numbers and the values differ slightly after the decimal point, treat it as a match with 4.
Return the Accuracy score in the following JSON format, with `Accuracy` and `Explanation` attribute:
```
{{
    "Accuracy": ... integer value between 0 - 5 ...,
    "Explanation": "... reason why you assigned the score ..."
}}
```
0 means the response is not useful at all and 5 means the response is the perfect answer to the question.


Evaluation: 
    """

    eval_template_v1 = """
[INST]
Please act as an impartial judge and evaluate the quality of the responses provided by an AI assistant to the user question displayed below.
User questions will ask for assistance on bugs, including summaries, explanations, analyses, or details beyond simple lists or counts.

### Definition
- **Query:** The original user question.
- **Query Category:** categories the input query belongs to. Options: List, Count, Single Bug Detailed, Other.
- **DB Query:** The AI assistant's SQL/Milvus query response.
- **Correct DB Query:** The expected ground truth SQL/Milvus query.
- **SQL Data:** The actual bug data used for verification.
- **Expected SQL Data:** The ground truth data expected as output.
- **Response:** Response from AI assistance.
- **Correct Answer:** Expected Response to be used as a reference.

### Query Category

- List: Queries that return a **list of bug IDs** only.
- Count: Queries that return a count of bugs. Note that the result of the query should be a single count. If a query returns multiple counts (like count of bugs filed by customer versus QA), it is not a count query. 
- Single Bug Detailed: Queries mentioning a specific bug ID and asking for detailed information.
            This category is only for queries asking about a single bug.
- Other: Queries that do not fit into any of the above categories.   



### Evaluation Process

Begin your evaluation by analyzing the response based on the provided elements. Provide a concise explanation of your judgment. Be objective and fair.

### Scoring Guide:

- **Disagree (Bad):** The response only partially meets the evaluation criteria and lacks acceptable quality.  
- **Agree (Good):** The response satisfies the criteria fully, provide results which aligns with Expected Result.  


### Evaluation Criteria:


**Accuracy (0-1):**  
- Has the assistant correctly categorized the query under one of the four intents (Count, List, Single Bug Detailed, Other).  
- Check if the response remain relevant to the user's intent.  
- Check if there are any syntax, logic, or output errors compared to the expected data.  
- For "Count" Category: Evaluate the numeric value of the answer given, comparing with the expected count.  
- For "List" Category: Evaluate the F1 score based on the precision and recall of the retrieved bug IDs compared to the expected list.  



### Output Format

Strictly follow the JSON schema to output your response.

{{
  "type": "object",
  "properties": {{
    "Explanation": {{
      "type": "string",
      "description": "A textual explanation or justification for the provided scores.",
    }},

    "Accuracy": {{
      "type": "integer",
      "minimum": 1,
      "maximum": 0
    }},
  }}
}}


### Important Guidelines:

- Avoid position biases and ensure response order does not influence your judgment.  
- Do not let response length influence your evaluation.  
- Be objective and impartial without favoring specific assistant names.  
- Ensure responses adhere to the specified format.

[/INST]

[INST]

Question: {question}

Assistant Answer: {answer}
    
{prompts_data}
"""