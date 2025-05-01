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


jobs_data = load_jobs()
if "job-titles" in jobs_data and isinstance(jobs_data["job-titles"], list):
    job_titles = jobs_data["job-titles"]
else:
    raise ValueError("The \"jobs.json\" file does not contain the expected key, or the key does not map to a list.")

embeddings = get_job_embeddings(job_titles)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))


def search_jobs(query, threshold=0.7):  # threshold küçükse daha yakın demek
    query_embedding = get_job_embeddings([query])
    query_embedding = np.array(query_embedding)

    # Tüm veri kümesine karşı arama yap
    D, I = index.search(query_embedding, len(job_titles))

    matched_jobs = [
        job_titles[i]
        for i, distance in zip(I[0], D[0])
        if distance < threshold
    ]
    return matched_jobs

