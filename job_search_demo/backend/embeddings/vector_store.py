import faiss
import numpy as np
import json
import os
from embeddings.embedder import get_job_embeddings


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS_PATH = os.path.join(BASE_DIR, "data", "jobs.json")

def load_jobs(path=JOBS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


job_titles = [title for title in load_jobs() if isinstance(title, str) and title.strip()]
embeddings = get_job_embeddings(job_titles)


dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))


def search_jobs(query, k=5):
    query_embedding = get_job_embeddings([query])
    D, I = index.search(np.array(query_embedding), k)
    return [job_titles[i] for i in I[0]]
