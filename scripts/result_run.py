import argparse

import requests
import json

url = 'https://devbot-api.nvidia.com/evaluation/run'


headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
run_type = "cron"
payloads = [
# {
#     "Project": "nvbot_for_nvhelp_mixtral_agent",
#     "RunType": run_type,
#     "System": "nvhelp",
#     "Model": "mixtral_agent",
#     "Attachments": [],
#     "UserId": "nvbot_evaluation"
# },
{
    "Project": "scout_mixtral_agent",
    "RunType": run_type,
    "System": "scout",
    "Model": "mixtral_agent",
    "UserId": "nvbot_evaluation"
},
# {
#     "Project": "avc_mixtral_sample",
#     "RunType": run_type,
#     "System": "avc",
#     "Model": "mixtral",
#     "Attachments": [],
#     "UserId": "nvbot_evaluation"
# }
]





if __name__ == '__main__':
    # epilog = "%s" % __Examples__
    parser = argparse.ArgumentParser(
        # epilog=epilog,
        description="Generate answers",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    print ("Start regression run")
    parser.add_argument("-local", help="Run triggered via localhost")
    # parser.add_argument("-userId", help="Nvidian Id")
    # parser.add_argument("-recipients", help="Result sent to email recipients")


    args = parser.parse_args()
    # NOTE: change port if needed
    local_port = args.local

    url = f"http://localhost:{local_port}/evaluation/batch_answer" if args.local else url

    for payload in payloads:
        print(url, json.dumps(payload))

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        print (response, response.status_code, response.text)
        if response.status_code == 200:
            print(f"{payload['Project']} request successful.")
            print("Response:", response.json())
        else:
            print(f"{payload['Project']} request failed.")
            print("Error:", response.status_code)

