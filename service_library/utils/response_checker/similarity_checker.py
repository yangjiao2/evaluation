from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score

from configs.settings import get_settings

# constants copied from langchain glean_constants.py
PRIVATE_RERANK_EMBEDDER = "nv-embedqa-mistral-7b-v2-prd"
PUBLIC_RERANK_EMBEDDER = "ai-nv-embedqa-mistral-7b-v2"
EMBEDDER_PAYLOAD_OVERRIDE = "nvidia/nv-embedqa-mistral-7b-v2"

def _nv_embedder_client_payload_fn(inputs):
    model_name = f"{EMBEDDER_PAYLOAD_OVERRIDE}"
    return {**inputs, "model": model_name}

def apply_cosine_similarities(list1, list2):
    nv_embedder = NVIDIAEmbeddings(model=PRIVATE_RERANK_EMBEDDER,
                                       api_key=get_settings().NVCF_API_KEY,
                                       truncate="END")

    nv_embedder.client.payload_fn = _nv_embedder_client_payload_fn
    list1_emb = nv_embedder.embed_documents(list1)
    list2_emb = nv_embedder.embed_documents(list2)
    cosine_similarities = cosine_similarity(list1_emb, list2_emb).tolist()
    return cosine_similarities

def apply_bert_f1(list1, list2):
    _, _, f1 = score(list1, list2, lang="en", model_type="bert-base-uncased", batch_size=64)

    return f1.tolist()
