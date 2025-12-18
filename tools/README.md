# Setup Port-forwarding
1. Setup aws credentials using [awsos-cli-token site](https://securityportal.nvidia.com/awsos-cli-token)
2. Check with `aws sts get-caller-identity` that you are authenticated after editing your `~/.aws/credentials` file
3. port forward eval-ms: `kubectl port-forward svc/evaluation-ms-prod-nemo-evaluator 7331 -n nemo-evaluation`
4. port forward mongodb: `kubectl port-forward svc/evaluation-ms-prod-nemo-evaluator 27017 -n nemo-evaluation`

# LLM Judge evaluation dataset & Custom evaluation dataset Validator
Use this tool to validate the format of the Judge files before submitting for evaluation jobs
simply `./check_dataset.sh` and provide the path to the folder in the UI

# Tooling: LLM Evaluation Monitoring Tooling
Use this tool to monitor the result of the evaluation jobs.
- install mongo client with: `pip install pymongo`
- Run with `streamlit run eval_browser.py`
Note: *require: port forwarding setup (see above)*


# Datastore Downloader
Use this tool to download datasets and evals from the nemo datastore (e.g.: https://datastore.stg.llm.ngc.nvidia.com/docs)

#### Requirements
1. connected to vpcs
2. `pip install huggingface-hub==0.20.3`

#### downloading eval or datasets
Tool to download e.g.

- dataset: `python datastore_hf_download.py dataset dataset-UEncBf2ZMuNJQPsFivXGwQ`
- eval: `python datastore_hf_download.py eval eval-UF3C7Pj4iN1xr7G6nGX29J`
