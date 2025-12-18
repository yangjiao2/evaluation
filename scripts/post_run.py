import argparse
import requests
import json


url = 'https://devbot-api.nvidia.com/evaluation/post_processing'


headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
# run_type = "daily_cron_job"
user_id = 'nvbot_evaluation'
# payloads = [
# {
#     "Project": "nvbot_for_nvhelp_mixtral_agent_sample",
#     "RunType": run_type,
#     "System": "nvhelp",
#     "Model": "mixtral_agent",
#     "Attachments": [],
#     "UserId": "nvbot_evaluation"
# },
# {
#     "Project": "avc_mixtral_sample",
#     "RunType": run_type,
#     "System": "avc",
#     "Model": "mixtral",
#     "Attachments": [],
#     "UserId": "nvbot_evaluation"
# }
# ]




if __name__ == '__main__':
    # epilog = "%s" % __Examples__
    parser = argparse.ArgumentParser(
        # epilog=epilog,
        description="Post processing Nemo Evaluation results",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    print ("Start post processing")
    parser.add_argument("-local", help="Run triggered via localhost")
    # parser.add_argument("-userId", help="Nvidian Id")
    # parser.add_argument("-recipients", help="Result sent to email recipients")

    args = parser.parse_args()
    local_port = args.local

    # NOTE: change port if needed
    url = f"http://localhost:{local_port}/evaluation/post_processing" if args.local else url
    payload = {
        "RunType": "cron",
        "CreatedByFrom": json.dumps({"days": -10}),
        "CreatedByTo": json.dumps({"days": 1})
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    print (response, response.status_code, response.text)
        # if response.status_code == 200:
        #     print(f"{payload['Project']} request successful.")
        #     print("Response:", response.json())
        # else:
        #     print(f"{payload['Project']} request failed.")
        #     print("Error:", response.status_code)
        #
