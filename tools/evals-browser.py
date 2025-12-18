import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
from browser_utils.connection import *


PAGE_SIZE = 20

# Function to fetch evaluations from MongoDB with pagination and filters
def fetch_evaluations(page_number, status_filter=None, evaluation_id_filter=None, type_filter=None):
    skip = page_number * PAGE_SIZE
    query = {}
    if status_filter:
        query["status"] = status_filter
    if evaluation_id_filter:
        query["_id"] = evaluation_id_filter
    if type_filter:
        query["evaluations.eval_type"] = type_filter
    result = list(collection.find(query,
        {"_id": 1, "status": 1, "model.llm_name": 1, "model.inference_url": 1, "evaluation_results": 1, "created_at": 1, "evaluations": 1})
                .sort("created_at", -1)
                .skip(skip)
                .limit(PAGE_SIZE))
    for eval in result:
        if "created_at" in eval:
            eval["created_at"] = eval["created_at"].strftime("%Y-%m-%d %H:%M:%S")
    return result

st.set_page_config(layout="wide")
# Streamlit UI
st.title("LLM Evaluation Monitoring Tool")

# Connection Status
st.subheader("Connection Status")
st.text("Note: This tool requires port-forwarding to the MongoDB (Eval). Please refer to the README for instructions.")
st.markdown(f"mongodb: `kubectl port-forward svc/evaluation-ms-prod-mongodb 27017 -n nemo-evaluation`")
st.markdown(f"gitea:  `kubectl port-forward svc/nemo-datastore-gitea-http 3000 -n nemo-evaluation`")
col1, col2, col3 = st.columns(3)
with col1:
    evaluation_ms_status = check_evaluation_ms()
    # st.write("Evaluation MS:", "🟢 Connected" if evaluation_ms_status else "🔴 Disconnected")
with col2:
    mongodb_status = check_mongodb()
    st.write("MongoDB:", "🟢 Connected" if mongodb_status else "🔴 Disconnected")
with col3:
    gitea_status = check_gitea_server()
    st.write("Gitea:", "🟢 Connected" if gitea_status else "🔴 Disconnected")

# Pagination state
if 'page_number' not in st.session_state:
    st.session_state.page_number = 0

# Fetch evaluations for the current page
evaluations = fetch_evaluations(st.session_state.page_number)

# Filters
st.subheader("Filters")
status_filter = st.selectbox("Status", options=["", "succeeded", "running", "failed"], index=0)
evaluation_id_filter = st.text_input("Evaluation ID")
type_filter = st.selectbox("Type", options=["", "llm_as_a_judge", "automatic"], index=0)

# Fetch evaluations for the current page with filters
evaluations = fetch_evaluations(st.session_state.page_number, status_filter, evaluation_id_filter, type_filter)

# Evaluations Table
st.subheader("Evaluations")

failed_ids = []

def redraw_evals():
    if evaluations:
        # Create DataFrame for the table
        eval_data = []
        for eval in evaluations:
            if eval["status"] == "failed":
                failed_ids.append(eval["_id"])
            created_at = eval["created_at"]
            eval_entry = {
                        "Created At": created_at,
                        "Evaluation ID": eval["_id"],
                        "Status": eval["status"],
                        # "Model Name": eval["model"]["llm_name"],
                        # "Inference URL": eval["model"]["inference_url"],
                    }
            # Extract input_dir or input_file based on eval_type
            for evaluation in eval.get("evaluations", []):
                eval_entry["type"] = evaluation.get("eval_type")
                if evaluation["eval_type"] == "llm_as_a_judge":
                    eval_entry["Input"] = evaluation.get("input_dir", "")
                elif evaluation["eval_type"] == "automatic":
                    eval_entry["Input"] = evaluation.get("input_file", "")

            # Extract scores from aggregated_results
            for result in eval.get("evaluation_results", []):
                for score in result.get("aggregated_results", []):
                    eval_entry[score["name"]] = score["value"]
                for result in result.get("task_results", []):
                    for m in result['metrics']:
                        eval_entry[f"{result['name']}.{m['name']}"] = m['value']
            eval_data.append(eval_entry)

        eval_df = pd.DataFrame(eval_data)
        def color_status_row(s):
            return ['background-color: red']*len(s) if s.Status == 'failed' else ['background-color: white']*len(s)

        st.dataframe(
            eval_df.style.apply(color_status_row, axis=1),
            column_config={
                "Details": st.column_config.Column(
                    "Details",
                    help=json.dumps(evaluations, indent=4),
                    width="medium",
                    required=True,
                )
            },
            hide_index=True
        )

        st.text("Note: to download evaluation for evaluation logs, see README#Datastore Downloader")

        # Detailed view buttons
        for eval in evaluations:
            col1, col2 = st.columns(2)
            _id = eval["_id"]
            failed = _id in failed_ids
            with col1:
                if st.button(f"Show Details for {_id}"):
                    st.json(json.dumps(eval, indent=4))
            with col2:
                t = "primary" if failed else "secondary"
                st.link_button(f"Gitea Repo: {_id}", f"http://localhost:3000/nvidia/{_id}", type=t)
    else:
        st.write("No evaluations found.")

# Pagination buttons
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.session_state.page_number > 0:
        if st.button("Previous Page"):
            st.session_state.page_number -= 1
with col3:
    if len(evaluations) == PAGE_SIZE:
        if st.button("Next Page"):
            st.session_state.page_number += 1

redraw_evals()
