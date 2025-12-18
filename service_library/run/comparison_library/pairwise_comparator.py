import datetime
import json
import logging
import re
import uuid
from typing import Any, List
import os
import numpy as np
from collections import Counter

from configs.settings import get_settings
from nvbot_utilities.utils.utilities import get_class_module, create_func_instance
from service_library.utils.data_helper import safe_json_loads
from service_library.utils.logging import log_errors
from bert_score import score

logger = logging.getLogger("Pairwise Comparator'")
from sklearn.metrics.pairwise import cosine_similarity


@log_errors('Generate Cosine Similarity from Embedding')
def generate_cosine_similarity_from_embedding(data_list: List[str]):

    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    NVCF_API_KEY = os.environ["NVIDIA_API_KEY"] #?
    embedder = NVIDIAEmbeddings(model="ai-embed-qa-4", truncate="END")
    embs = embedder.embed_documents(data_list)  # example: queries are a list of queries

    q_scores = cosine_similarity(embs, embs)
    avg = round(np.mean(q_scores), 4)
    var = round(np.var(q_scores), 4)

    return q_scores, avg, var


@log_errors('Generate Cosine Similarity from f1')
def generate_cosine_similarity_from_f1(data_list: List[str]):
    n = len(data_list)
    similarity_matrix = np.zeros((n, n))

    # Calculate pairwise scores
    for i in range(n):
        for j in range(i, n):  # Avoid redundant calculations
            if i == j:
                similarity_matrix[i][j] = 1.0  # Similarity with itself is 1
            else:
                # Compute BERT score for pair (strings[i], strings[j])
                P, R, F1 = score([data_list[i]] or "", [data_list[j]] or "", lang='en', model_type="bert-base-uncased", batch_size=32)
                similarity = F1.mean().item()  # Use F1 score as similarity
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity  # Symmetric matrix

    # Extract upper triangular part, excluding diagonal
    non_diagonal_elements = similarity_matrix[np.triu_indices(n, k=1)]
    avg = round(np.mean(non_diagonal_elements), 5)
    var = round(np.var(non_diagonal_elements), 5)

    return similarity_matrix, avg, var

# take scout queries, 5 runs, output of bert (matrix mean)
# pick variance of length of query and response
# 3 scout_v1 vs. scout_v2

def derive_key_valuelist_from_dicts_list(json_strings):
    # print ("json_strings:", json_strings)
    dicts = [safe_json_loads(item, {}) for item in json_strings]

    # Collect all unique keys from the dictionaries
    key_sets = set()
    for d in dicts:
        if isinstance(d, dict):
            key_sets.update(d.keys())

    # Initialize the resulting dictionary with keys and empty lists
    result = {key: [] for key in key_sets}

    for key in key_sets:
        for d in dicts:
            result[key].append(str(d.get(key, "")) if isinstance(d, dict) else "")

    return result

@log_errors('Find Outliers')
def find_outliers_from_cosine_similarity_metrics(metrics, column_name: str, threshold: float =0.7, ):
    """
        Identifies outliers in a cosine similarity matrix.

        Parameters:
            metrics (ndarray): A square matrix of cosine similarity metrics.
            threshold (float): The threshold below which a value is considered an outlier.

        Returns:
            tuple: A tuple containing a list of outlier pairs and the most frequent outlier.
        """
    n = metrics.shape[0]
    # Get pairs of indices for values below the threshold
    outliers = []
    for i in range(n):
        for j in range(i + 1, n):  # Only consider upper triangular part (excluding diagonal)
            if metrics[i, j] < threshold:
                outliers.append((i + 1, j + 1))  # Use 1-based indexing

    # Flatten outliers to find the most frequent outlier
    flat_outliers = [idx for pair in outliers for idx in pair]
    outlier_counter = Counter(flat_outliers)
    indexes = []
    if outlier_counter:
        max_frequency = max(outlier_counter.values())
        most_frequent_outlier = [
            f"{column_name}.{idx}" for idx, freq in outlier_counter.items() if freq == max_frequency
        ]
        indexes = [
            idx for idx, freq in outlier_counter.items() if freq == max_frequency
        ]
    else:
        most_frequent_outlier = []

    return outliers, most_frequent_outlier, indexes

