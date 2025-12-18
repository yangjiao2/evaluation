import sys
import huggingface_hub as hh

eg_dataset_name = "nvbot_evaluation-nvbot_for_nvhelp_mixtral_agent_sample2-llm_as_a_judge-0604-1003-PWYmdsZ"
eg_eval_name = "eval-T6MbSvLsUS9iksemcuAf2M"

###
def download_dataset(dataset_name):
    repo_name = dataset_name
    ds_url = "https://datastore.stg.llm.ngc.nvidia.com"
    token = "mock"
    download_path = f"datasets/{repo_name}"
    repo_name = f'nvidia/{repo_name}'

    api = hh.HfApi(endpoint=ds_url, token=token)
    repo_type = 'dataset'
    api.snapshot_download(repo_id=repo_name, repo_type=repo_type, local_dir=download_path, local_dir_use_symlinks=False)
    print(f"Dataset downloaded to {download_path}")

def download_evals(eval_name):
    repo_name = eval_name
    ds_url = "https://datastore.stg.llm.ngc.nvidia.com"
    token = "mock"
    download_path = f"evals/{repo_name}"
    repo_name = f'nvidia/{repo_name}'

    api = hh.HfApi(endpoint=ds_url, token=token)
    repo_type = 'dataset'  # type eval not available yet on staging data store
    api.snapshot_download(repo_id=repo_name, repo_type=repo_type, local_dir=download_path, local_dir_use_symlinks=False)
    print(f"Evaluation downloaded to {download_path}")

if __name__ == "__main__":
    # download_dataset()
    # check if at least 2 argument is provided
    # if not, print the usage of the script
    print(f"Number of arguments: {len(sys.argv)}")
    if len(sys.argv) != 3:
        print("Usage: python datastore_hf_download.py [dataset|eval] [dataset_name/eval_name]")
        print("E.g: python datastore_hf_download.py eval eval-Y6F3kHqzcJzNGqYd1SfGmo")
        print("E.g: python datastore_hf_download.py dataset eval-Y6F3kHqzcJzNGqYd1SfGmo")
        sys.exit(1)
    type = sys.argv[1]
    name = sys.argv[2]
    if type == 'dataset':
        download_dataset(name)
    elif type == 'eval':
        download_evals(name)
