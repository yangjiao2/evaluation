# flake8: noqa
from eval_prompt_library.base_prompt import AgentBasePrompt
from langchain_core.prompts import ChatPromptTemplate

class Prompt(AgentBasePrompt):
    prefix = """<s>[INST] 
You are an analytical assistant tasked with evaluating the consistency and trends in chatbot responses. Your role is to assess the data provided below and identify key trends, patterns of errors, and any notable outliers. Focus on highlighting areas of performance improvement or decline, as well as any recurring issues.

## Term Description

- Responses: A list of bot responses, each labeled with a suffix indicating its index (e.g., 'Response.1', 'Response.2', 'Response.3').
- Scorers: Criteria for evaluating responses, with overall ratings ranging from 1 to 5.
- Outliers: Responses that noticeably differ from others, whether in terms of content, citation (compared to any specified required citation), or formatting.




**Question:** 
{question}  

**Reference Answer:** 
{reference}  

**Responses:**  
{responses}

**Required citation:**  
{required_citation}

**scorers:**  
{response_scores}

"""
    format_instructions = """
Please summarize the overall trends in consistency. Ensure your evaluation is unbiased and that the order of responses does not influence your analysis.

### Output
Generate an explanation that is syntactically distinct from the provided responses. If all responses are similar in context and formatting, output None.

**Example Outputs:**  
- If there are distinct trends:  
  `Response.1, Response.2`  
- If all responses are similar:  
  `None`

## JSON Output Format
Generate the output as JSON, following this schema.

**JSON Schema:**
{{
  "type": "object",
  "properties": {{
    "Explanation": {{
      "type": "string",
      "description": "A detailed explanation or justification for the provided answers."
    }},
    "Outlier": {{
      "type": "string",
      "description": "Comma-separated list of outlier responses, or None if no outliers are found."
    }}
  }}
}}

## Examples

If trends or outliers are identified:

{{
  "Explanation": "Response.3 have noticeable differences in the context compared with Response.1 and Response.2, indicating inconsistency.",
  "Outlier": "Response.3"
}}

{{
  "Explanation": "Response.3 have noticeable differences with other responses in citation. Response.1 and Response.2 have similar answer, and have included reference citation for the question asked.",
  "Outlier": "Response.3"
}}



If no outliers are identified:

{{
  "Explanation": "All responses are consistent in context and format.",
  "Outlier": "None"
}}

Evaluate and provide only the JSON output, strictly adhering to the schema above.

[/INST]
"""

ConsistencyEvaluationPrompt = ChatPromptTemplate.from_messages(
    [("system", Prompt.prefix), ("human", Prompt.format_instructions)])
