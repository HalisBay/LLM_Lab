from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")

def get_job_embeddings(job_titles):
    embeddings = model.encode(job_titles, show_progress_bar=True)
    return embeddings
