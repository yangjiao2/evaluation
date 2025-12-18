create_query_variation_prompt = """
Given below QUESTION, please create five query variation of this question which keep the same meaning and intent.
- Each of the query variation must be about the same length of the QUESTION.
- Your output must be a Python List format as below:
```
["variation_1", "variation_2", "variation_3", "variation_4", "variation_5"]
```


QUESTION: {question}
"""

generalize_response_prompt = """
Below RESPONSE is a LLM generated answer for the QUESTION.
The RESPONSE may contain some personal identity information like user name, etc.
Your task is to remove such personal identity information from the RESPONSE to output a general answer to the QUESTION for everyone.

Your OUTPUT should strictly follow below rule:
- The OUTPUT should be almost the same as RESPONSE, using same wordings as much as possible.
- The OUTPUT should only remove user name.
- If there are URLs, links, or inline citation in MARKDOWN [Title](URL) format, you MUST keep them as is.
- If there is long single paragraph in RESPONSE, try to separate them into shorter ones without modifying the information.
- You MUST ONLY output the new response body itself WITHOUT any explanation or starter message.


QUESTION: {question}


RESPONSE:
```
{response}
```
"""