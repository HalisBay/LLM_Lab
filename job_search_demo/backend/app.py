from fastapi import FastAPI
from embeddings.embedder import get_job_embeddings
from embeddings.vector_store import search_jobs

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/search/")
async def search_jobs_endpoint(query: str):
    results = search_jobs(query)
    return {"results": results}

@app.get("/health/")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
