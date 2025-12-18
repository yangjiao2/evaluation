import json
import re
from typing import Any

from service_library.utils.logging import log_errors

import json
import math
from typing import Any

from service_library.nemo_ms.nemo_service_helper import convert_number
from service_library.utils.logging import log_errors

VA_BASE_URL = "https://nvidia.service-now.com"


@log_errors("VA response")
def extract_va_response(data):
    result = data.get("Result", {})
    response = result.get("conversation_response", {})
    answer = response.get("value", "")
    citations = response.get("citations", [])

    response_citation = []
    full_answer = answer
    for citation in citations:
        response_citation.append(citation.get("citationHref", ""))
        # if citation.get("table") == "kb_knowledge":
        #     response_citation.append(
        #         f"[{citation.get('short_description', '')}]({VA_BASE_URL}{citation.get('url', '')})"
        #     )
        # elif citation.get("table") == "sc_cat_item":
        #     # full_answer += f"\n{citation.get('name', '')}\n{citation.get('description', '')}"
        #     response_citation.append(
        #         f"[{citation.get('name', '')}]({VA_BASE_URL}{citation.get('url', '')})"
        #     )
        # else:
        #     response_citation.append(
        #         f"[{citation.get('short_description', '')}]({citation.get('url', '')})"
        #     )

    if not full_answer:
        return {"Response": "", "Answer": "", "Citations": ""}

    return {
        "Response": full_answer + "\n\n" + "\n".join(response_citation),
        "Answer": answer,
        "Citations": response_citation,
    }


# Sample JSON data
data1 = {
    "result": {
        "original_utterance": "what is a spam and how to avoid it?",
        "conversation_response": {
            "answer": 'Spam refers to unsolicited and unwanted emails, often sent to sell products or services. To avoid spam, check your Outlook Junk folder for suspicious emails. You can report spam by using the "Report Message" button in Outlook and selecting "Junk." This action moves the email to your Junk folder and blocks the sender. If a legitimate email is marked as spam, you can use the "Not Junk" option to move it back to your Inbox and add the sender to your Safe Senders list ​[1]​.',
            "format": "markdown",
            "citations": [
                {
                    "table": "kb_knowledge",
                    "short_description": "FAQ-Spam & Phishing Emails",
                    "sys_id": "985c2dccc3c696904591d4af05013101",
                    "url": "/esc?id=kb_article&sys_id=985c2dccc3c696904591d4af05013101",
                }
            ],
        },
    }
}

data2 = {
    "result": {
        "original_utterance": "I want to request headset",
        "conversation_response": {
            "answer": 'You can request a new or replacement headset by using the "Computer Accessory Request" option ​[1]​.',
            "format": "markdown",
            "citations": [
                {
                    "name": "Computer Accessory Request",
                    "description": "<p>Request a new or replacement accessory for your laptop or desktop PC. E.g. Camera, Mouse, Keyboard, Docking Stations, Cables, Headsets etc...</p>",
                    "sys_id": "43f82e7d97fc1590275ef7300153af21",
                    "table": "sc_cat_item",
                    "url": "/esc?id=sc_cat_item&sys_id=43f82e7d97fc1590275ef7300153af21",
                }
            ],
        },
    }
}


#
# # Run function
# print(json.dumps(extract_response(data1), indent=2))
# print(json.dumps(extract_response(data2), indent=2))


def extract_citations(text):
    # Define URL pattern
    url_pattern = re.compile(r"https?://[^\s\)\]'\"]+")

    # Find all URLs in the text
    urls = url_pattern.findall(text)

    # Patterns to extract KB and catalog sys_id
    # https://nvidiadev.service-now.com/esc?id=kb_article&sys_id=2df518bf839a52107e5b7b226daad3cc
    kb_pattern = re.compile(
        r"(?:sysparm_article=(KB\d+)|kb_article&sys_id=([a-f0-9]{30,}))"
    )
    catalog_pattern = re.compile(r"sys_id=([a-f0-9]{30,})")

    url_list = []
    for url in urls:
        kb_url, catalog_url = kb_pattern.findall(url), catalog_pattern.findall(url)
        if len(kb_url) == 0 and len(catalog_url) == 0:
            url_list.append(url.strip())
        url_list.extend(kb_pattern.findall(url.strip()))
        url_list.extend(catalog_pattern.findall(url.strip()))

    return url_list


@log_errors("va citation score")
def citation_score(input_dict: Any):
    values = input_dict.get("values")
    row = input_dict.get("row")
    try:
        if convert_number(row.get("Status Code")) != 200:
            return {}

        def is_null_like(value):
            return (
                value is None
                or value == ""
                or (isinstance(value, float) and math.isnan(value))
            )

        required_citations = (
            extract_citations(row.get("Required Citations"))
            if not is_null_like(row.get("Required Citations"))
            else []
        )
        retrieved_citations = (
            extract_citations(row.get("Citations", []))
            if not is_null_like(row.get("Citations"))
            else []
        )

        citation_correctness_score = 3
        retrieved_set = set(retrieved_citations)
        ground_truth_set = set(required_citations)
        if len(ground_truth_set) != 0:

            relevant_retrieved = retrieved_set.intersection(ground_truth_set)
            recall = len(relevant_retrieved) / len(ground_truth_set)
            if recall == 0:
                citation_correctness_score = 0
            elif (recall - 1) < 1e-5 and len(ground_truth_set) == len(retrieved_set):
                citation_correctness_score = 5

        else:
            citation_correctness_score = 5 if len(retrieved_set) == 0 else 3

        # Compute Overall Score
        judge_correctness = convert_number(values.get("Correctness Answer", 0))
        if judge_correctness in [4, 5] and citation_correctness_score == 0:
            overall_score = 3
        elif citation_correctness_score == 5:
            overall_score = judge_correctness
        elif judge_correctness in [4, 5] and citation_correctness_score == 3:
            overall_score = judge_correctness - 1
        else:
            overall_score = judge_correctness

        row["Expected Citations"] = required_citations
        row["Retrieved Citations"] = retrieved_citations
        row["Citation Correctness"] = citation_correctness_score
        row["Overall"] = overall_score

        return {"row": row, "values": values}
    except Exception as e:
        print(f"Error updating citation score from {json.dumps(input_dict)}: {e}")
        return values


if __name__ == "__main__":

    citation_correctness_score = 3
    if len(["a"]) != 0:
        retrieved_set = set(["879a291c9739a590275ef7300153af87"])
        ground_truth_set = set(["879a291c9739a590275ef7300153af87"])
        relevant_retrieved = retrieved_set.intersection(ground_truth_set)
        recall = len(relevant_retrieved) / len(ground_truth_set)
        if recall == 0:
            citation_correctness_score = 0
        elif (recall - 1) < 1e-5 and len(ground_truth_set) == len(retrieved_set):
            citation_correctness_score = 5
        # print (citation_correctness_score)
