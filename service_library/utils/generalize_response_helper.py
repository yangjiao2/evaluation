import ast

from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from configs.settings import get_settings
from eval_prompt_library.query_generalization_prompt import generalize_response_prompt, create_query_variation_prompt
from service_library.run.parser_library.response_parser import SCOUT_CITATION_HEADER, NVBOT_CITATION_HEADER


def generalize_response(question, response_text):
    # print ("Question", question)
    # print ("Response", response_text)
    content, keyword_used, citation = response_text.partition(SCOUT_CITATION_HEADER)
    if not keyword_used:
        content, keyword_used, citation = response_text.partition(NVBOT_CITATION_HEADER)

    template = ChatPromptTemplate.from_messages([
        ("human", generalize_response_prompt),
    ])

    args = {
        "question": question,
        "response": content
    }
    prompt_value = template.invoke(args)
    llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1", temperature=0.0, max_token=500, seed=42,
                     api_key=get_settings().PUBLIC_NVCF_API_KEY)
    result = llm.invoke(prompt_value)
    generalized_response = result.content
    # print("\ngeneralized_response:", generalized_response)
    # print("\ncitation:", citation)
    return f"{generalized_response}\n\n{keyword_used}{citation}"


def create_query_variation(query, response):
    template = ChatPromptTemplate.from_messages([
        ("human", create_query_variation_prompt),
    ])

    args = {
            "question": query,
            "response": response
        }
    prompt_value = template.invoke(args)
    llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1", temperature=0.0, max_token=500, seed=42,
                     api_key=get_settings().PUBLIC_NVCF_API_KEY)
    result = llm.invoke(prompt_value)
    queries = ast.literal_eval(result.content)
    return queries