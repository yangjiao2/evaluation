IRQA_DOMAIN = "irqa"
IRQA_INTENT = "irqa"
DOMAIN_ENDPOINT_MAP = {
    "gitlab": "gitlab",
    "jira": "jira",
    "it_service_management": "itservicemanagement",
    "nvbugs": "nvbugs",
}
MULESOFT_HEADER = {
    "client_id": "",
    "client_secret": "",
    "Content-Type": "",
}


REGRESSION_S3_DATASET_PATH = "dataset/"
REGRESSION_TMP_DATASET_FILE = "tmp_datafile.xlsx"
REGRESSION_S3_RESULTS_PATH = "results/"
REGRESSION_S3_DAILY_RUN_PATH = "daily_runs/"
REGRESSION_TMP_RESULT_FILE = "tmp_result.xlsx"
REGRESSION_S3_BUCKET = "llm-evaluation"
REGRESSION_SERVER_URL = "http://127.0.0.1:5000/langchain/agent"
SNOW_INC_URL = "https://nvidia.service-now.com/api/now/table/incident?"
SNOW_INC_PARAM = "sysparm_query=number%3D{}&sysparm_display_value=true&sysparm_exclude_reference_link=true"
SNOW_INC_FIELDS = "&sysparm_fields=u_reopened%2Cu_escalation_reason%2Cnumber%2Cstate%2Cimpact%2Cpriority\
    %2Cu_on_behalf_of%2Cshort_description%2Creassignment_count%2Ccomments_and_work_notes%2Csys_created_on\
    %2Ccategory%2Creopen_count%2Curgency%2Cclose_notes%2Cdescription%2Csubcategory%2Copened_at\
    %2Creopened_time%2Cresolved_at"
SNOW_SR_URL = "https://nvidia.service-now.com/api/now/table/sc_req_item?"
SNOW_SR_PARAM = "sysparm_query=number%3D{}&sysparm_display_value=true&sysparm_exclude_reference_link=true"
SNOW_SR_FIELDS = "&sysparm_fields=number%2Crequested_for%2Cimpact%2Cpriority%2Curgency%2Csys_created_on%2Cclosed_at\
    %2Cshort_description%2Cdescription%2Creassignment_count%2Ccomments_and_work_notes%2Cclose_notes%2Cstate"
GLEAN_CHAT_BASELINE = "run_glean_chat_full_20231013_fixed.xlsx"
SME_GT_BASELINE_FILE = ""
SME_GT_PATH = "groundtruth/"

STATUS_START = "status_update_start"
STATUS_END = "status_update_done"
LLM_UNAVAILABLE = "I was unable to communicate with NVIDIA AI Playground.  Please try again later."
STATUS_MESSAGE = "Combing through NVIDIA Blogs, press releases & earnings announcements…\n"
DATE_CONTEXT_STR = "- The Current Date is {}. The upcoming days includes {} (days after {}), {}, {}, and {}."
DATE_CONTEXT_STR_V2 = "Note: The Current Date is year: {} month: {}."
DATE_CONTEXT_STR_V3 = "- The Current Date is {} Pacific Time."
# DATE_CONTEXT_STR = "Note: Today's date is {} Pacific Time. \
#                     You MUST consider today's date context when answering user's question."
# TODO Aaditya plz check me here - Sean ^
APPLICATION_EXCEPTION = "I seem to be having a problem. Please try again in a while."
DATE_CONTEXT_FINANCE_STR = "It is the calendar year {}, and we are in Q{} of the fiscal year {}."
MAX_PROMPT_WORDS_COUNT = 2000
NVIDIA_PLAYGROUND_MIXTRAL_IT = "playground_mixtral_8x7b"
HTML_CHUNKS = 10
KW_CHUNKS = 10
SEC_CHUNKS = 2
JHH_CHUNKS = 2
FISCAL_YEAR_PREV = "fiscal_year_prev"
FISCAL_QUARTER_PREV = "fiscal_quarter_prev"
FISCAL_YEAR_NEXT = "fiscal_year_next"
FISCAL_QUARTER_NEXT = "fiscal_quarter_next"
