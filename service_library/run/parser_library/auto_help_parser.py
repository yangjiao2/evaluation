import re
from service_library.utils.logging import log_errors

@log_errors("Auto Help response")
def extract_auto_help_response(data):
    llm_response = ""
    documents = ""
    response_status = "incomplete"
    try:
        if data:
            llm_response = str(data["result"]["skillResponse"])
            response_status = str(data["result"]["result"])
            documents = str(data["result"]["documents"])
            rephrasal = str(data["result"]["rephrasal"])
            # Remove [code] and [/code] using regex
            llm_response = re.sub(r"\[/*code\]", "", llm_response)
    except Exception as err:
        print(f"Cant process skill output {data}: {err}")

    return {
        "Response": llm_response,
        "Response Status": response_status,
        "Documents": documents,
        "Rephrasalß": rephrasal
    }


# Sample JSON data
data1 = {
    "result": {
        "result": "completed",
        "skillResponse": "[code]\nPlease submit a request for a loaner laptop using the [Loaner Asset Request form](https://nvidiadev.service-now.com/com.glideapp.servicecatalog_cat_item_view.do?v=1&sysparm_id=255e97b1736610107e88ef66fbf6a7e3). Complete the required fields to ensure your request is processed. Loaner laptops are issued for a two-week period, with an option to extend based on availability.\n\nCitations:\n- [Loaner Devices (Laptops, Desktops, & Monitors)](https://nvidiadev.service-now.com/kb_view.do?sys_kb_id=000f0737871b16506d03c8c7dabb35bd)\n- [Loaner Asset Request](https://nvidiadev.service-now.com/com.glideapp.servicecatalog_cat_item_view.do?v=1&sysparm_id=255e97b1736610107e88ef66fbf6a7e3)\n[/code]"
    }
}
