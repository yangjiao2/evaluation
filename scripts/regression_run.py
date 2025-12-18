import argparse

import requests
import json

url = 'https://devbot-api.nvidia.com/evaluation/eval_run'


headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
run_type = "daily_cron_job"
payloads = [
    {
    "ProjectId": 1,
    "Project": "nvbot_for_nvhelp_mixtral_agent_e2e_sample",
    "RunType": "cron",
    "System": "nvhelp",
    "Model": "mixtral_agent",
    "Attachments": [],
    "UserId": "nvbot_evaluation"
},

]


if __name__ == '__main__':
    # epilog = "%s" % __Examples__
    parser = argparse.ArgumentParser(
        # epilog=epilog,
        description="launch Nemo Evaluation custom run",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    print ("Start Nemo Evaluator run")
    parser.add_argument("-local", help="Run triggered via localhost")
    # parser.add_argument("-userId", help="Nvidian Id")

    args = parser.parse_args()
    # NOTE: change port if needed
    local_port = args.local
    url = f"http://localhost:{local_port}/evaluation/eval_run_e2e" if args.local else url
    print ('local', args.local)
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

