from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    Document,
    PromptTemplate,
)

from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

import qdrant_client
from pathlib import Path

# ==========================================================
# LLM + Embedding
# ==========================================================

llm = Ollama(model="qwen2.5:0.5b", request_timeout=120.0)

embed_model = FastEmbedEmbedding(model_name="BAAI/bge-large-en-v1.5")

rerank = SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=2)

Settings.llm = llm
Settings.embed_model = embed_model

# ==========================================================
# Load Docs
# ==========================================================


def load_docs(doc_path="./docs"):
    docs = []

    txt_paths = sorted(Path(doc_path).rglob("*.txt"))

    for path in txt_paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            continue

        docs.append(
            Document(
                text=text,
                metadata={
                    "file_name": path.name,
                    "file_path": str(path),
                },
            )
        )

    return docs


docs = load_docs()

# ==========================================================
# Qdrant Setup
# ==========================================================

client = qdrant_client.QdrantClient(host="localhost", port=6333, timeout=120.0)

collection_name = "document_chat"

vector_store = QdrantVectorStore(
    client=client, collection_name=collection_name, prefer_grpc=False
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ==========================================================
# INDEX BUILD / REUSE LOGIC
# ==========================================================


collection_exists = client.collection_exists(collection_name=collection_name)


if collection_exists:
    print(f"[INFO] Using existing Qdrant collection: {collection_name}")

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

else:
    print(f"[INFO] Building new index: {collection_name}")

    if not docs:
        raise ValueError("No documents found!")

    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

# ==========================================================
# Query Engine
# ==========================================================

query_engine = index.as_query_engine(similarity_top_k=2, node_postprocessors=[rerank])

template = """
Context:
---------------------
{context_str}
---------------------

Answer the question clearly.

Question: {query_str}

Answer:
"""

query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": PromptTemplate(template)}
)


def get_query_engine():
    return query_engine
