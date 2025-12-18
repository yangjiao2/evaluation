import os
import docker
import logging
import tarfile
import pandas as pd
import glob
import datetime

from service_library.constants import LOCAL_EVAL_RESULTS_TMP_FOLDER

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Variables ---
TIMESTAMP = datetime.datetime.now().strftime("%m%d_%H-%M")
RESULTS_DIR = os.path.abspath(f"{LOCAL_EVAL_RESULTS_TMP_FOLDER}/genai-perf-results_{TIMESTAMP}")

SWEEP_RANGE = "1:16"
TOKENIZER = "hf-internal-testing/llama-tokenizer"

# --- Test Configurations ---

CONFIGURATIONS = [
    {
        "name": "kv_cache_llama3.1_70b_instruct",
        "enabled": True,
        "model": "meta/llama-3.1-70b-instruct",
        "tag": "kv_cache_deterministic",
        "url_template": "`https://datarobot.stg.astra.nvidia.com/api/v2/deployments/6882b3b9303de70bd9ca8d85`",
        "endpoint": None,
        "auth_token_env": "GENAI_TOKEN",
        "auth_token_default": "..",
        "tokenizer": "hf-internal-testing/llama-tokenizer",
        "input_means": [1000, 5000, 10000, 20000],
        "output_mean": [1000, 1000, 1000, 1000],
        "streaming": True,
    },
    {
        "name": "datarobot_llama3.1_70b_instruct",
        "enabled": True,
        "model": "meta/llama-3.1-70b-instruct",
        "tag": "684c699791c7b7bb04a9bcf4",
        "url_template": "https://datarobot.stg.astra.nvidia.com/api/v2/deployments/6870131a5513f63b746e7465",
        "endpoint": None,
        "auth_token_env": "GENAI_TOKEN",
        "auth_token_default": "..",
        "tokenizer": "hf-internal-testing/llama-tokenizer",
        "input_means": [1000, 5000, 10000, 20000],
        "output_mean": [1000, 1000, 1000, 1000],
        "streaming": True,
    },
    {
        "name": "nvcf_llama3.1_70b_instruct",
        "enabled": True,
        "model": "nvdev/it-ent-ai/llama-3.1-70b-instruct",
        "tag": "NVDev",
        "url_template": "https://api.nvcf.nvidia.com/v2",
        "endpoint": "nvcf/pexec/functions/92b03c91-39ed-4b19-9560-53c9fee1f08c",
        "auth_token_env": "NVCF_TOKEN",
        "auth_token_default": "nvap-..",
        "tokenizer": "hf-internal-testing/llama-tokenizer",
        "input_means": [1000, 5000, 10000, 20000],
        "output_mean": [1000, 1000, 1000, 1000],
        "streaming": False,
    }
]


def run_perf_test(client, image, config):
    """Runs a single genai-perf test based on a configuration dictionary."""
    tag = config["tag"]
    container_name = f"perf_{tag}"
    logging.info(f"--- Starting test: {config['name']} ---")

    # Clean up any old container with the same name
    try:
        old_container = client.containers.get(container_name)
        logging.warning(f"Removing existing container '{container_name}'...")
        old_container.remove(force=True)
    except docker.errors.NotFound:
        pass  # No old container to remove

    container = None
    try:
        # 1️⃣ Start a reusable container
        logging.info(f"Starting container '{container_name}'...")
        container = client.containers.run(
            image,
            command="sleep infinity",
            detach=True,
            network_mode="host",
            name=container_name,
        )
        logging.info(f"✅ Container '{container_name}' is running.")

        auth_token = os.getenv(config["auth_token_env"], config["auth_token_default"])
        auth_header = f"Authorization: Bearer {auth_token}"
        url = config["url_template"]

        for i, inp in enumerate(config["input_means"]):
            host_dir = os.path.join(RESULTS_DIR, config['name'], f"in_{inp}")
            os.makedirs(host_dir, exist_ok=True)
            export_path = f"{tag}_input_{inp}_analyze.json"

            cmd = [
                "genai-perf", "analyze",
                "-m", config["model"],
                "--endpoint-type", "chat",
                "--url", url,
                "--tokenizer", config.get("tokenizer", TOKENIZER),
                "--synthetic-input-tokens-mean", str(inp),
                "--synthetic-input-tokens-stddev", "100",
                "--output-tokens-mean", str(config["output_mean"][i] if len(config["output_mean"]) >= i else str(inp)),
                "--sweep-type", "concurrency",
                "--sweep-range", str(config.get("sweep-range") or SWEEP_RANGE),
                "--profile-export-file", export_path,
                "--generate-plots",
                "--artifact-dir", "/artifacts",
                "--header", auth_header
            ]

            if config.get("endpoint"):
                cmd.extend(["--endpoint", config["endpoint"]])
            if config.get("streaming"):
                cmd.append("--streaming")

            logging.info(f"▶ Running analyze for input tokens = {inp}")
            exit_code, output = container.exec_run(cmd)

            if exit_code != 0:
                logging.error(f"❌ Error (exit {exit_code}) in test '{config['name']}':\n{output.decode()}")
                break

            logging.info("📦 Retrieving artifacts...")
            bits, _ = container.get_archive("/artifacts")
            tar_path = os.path.join(host_dir, f"artifacts_{inp}.tar")
            with open(tar_path, "wb") as f:
                for chunk in bits:
                    f.write(chunk)
            logging.info(f"✅ Artifacts for INPUT={inp} saved to {tar_path}")

            # Unpack the artifacts
            logging.info(f"Unpacking artifacts from {tar_path}...")
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(path=host_dir)
            logging.info(f"✅ Artifacts unpacked in {host_dir}")

    except docker.errors.APIError as e:
        logging.error(f"Docker API Error during test '{config['name']}': {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during test '{config['name']}': {e}")
    finally:
        if container:
            logging.info(f"🧹 Stopping and removing container '{container_name}'...")
            container.remove(force=True)
            logging.info("✅ Container removed.")
    logging.info(f"--- Finished test: {config['name']} ---")


def start(configurations):
    """Main function to run all enabled genai-perf tests."""
    image = "nvcr.io/nvidia/tritonserver:25.06-py3-sdk"
    client = docker.from_env()

    try:
        logging.info(f"Pulling Docker image '{image}'...")
        client.images.pull(image)
        logging.info("✅ Image pulled successfully.")
    except docker.errors.APIError as e:
        logging.error(f"Failed to pull image '{image}'. Please ensure Docker is running. Error: {e}")
        return

    for config in configurations:
        if config.get("enabled", False):
            run_perf_test(client, image, config)
        else:
            logging.warning(f"Skipping disabled test: {config['name']}")

    # After all tests, collect the results
    collect_and_save_results()


def collect_and_save_results():
    """Collects all analyze_export_genai_perf.csv files and saves them to an Excel spreadsheet."""
    logging.info("--- Collecting all results into a single spreadsheet ---")
    results_dir = RESULTS_DIR
    search_pattern = os.path.join(results_dir, "**", "analyze_export_genai_perf.csv")
    csv_files = glob.glob(search_pattern, recursive=True)

    if not csv_files:
        logging.warning("No 'analyze_export_genai_perf.csv' files found to consolidate.")
        return

    excel_path = os.path.join(results_dir, "genai_perf_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for csv_file in csv_files:
            try:
                # Extract a concise and valid sheet name from the path
                try:
                    parts = csv_file.split(os.sep)
                    config_name = parts[-4]
                    input_size_name = parts[-3]

                    # Create a shorter name like "astronomy_in_1000"
                    short_config = config_name.split('_')[0]
                    sheet_name = f"{short_config}_{input_size_name}"

                    # Truncate to Excel's 31-character limit as a safeguard
                    if len(sheet_name) > 31:
                        sheet_name = sheet_name[:31]
                except (IndexError, TypeError):
                    # Fallback sheet name if path structure is unexpected
                    sheet_name = os.path.basename(os.path.dirname(csv_file))

                df = pd.read_csv(csv_file)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                logging.info(f"✅ Added '{csv_file}' to sheet '{sheet_name}' in '{excel_path}'")
            except Exception as e:
                logging.error(f"❌ Failed to process '{csv_file}': {e}")

    logging.info(f"--- ✅ All results saved to '{excel_path}' ---")


if __name__ == "__main__":
    start(CONFIGURATIONS)
