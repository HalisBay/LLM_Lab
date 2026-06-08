from __future__ import annotations

import argparse
import os
from pathlib import Path

import nest_asyncio

nest_asyncio.apply()

from llama_index.core import PropertyGraphIndex, Settings, SimpleDirectoryReader
from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    SimpleLLMPathExtractor,
    VectorContextRetriever,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.ollama import Ollama

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dbs"
HF_CACHE_DIR = BASE_DIR / "hf_cache"
DEFAULT_QUERY = "What does YC do?"
DEFAULT_OLLAMA_MODEL = "qwen2.5:0.5b"
DEFAULT_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_NEO4J_URL = "neo4j://localhost:7687"
DEFAULT_NEO4J_USERNAME = "neo4j"
DEFAULT_NEO4J_PASSWORD = "password"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and query a Neo4j-backed GraphRAG index."
    )
    parser.add_argument(
        "--query", default=DEFAULT_QUERY, help="Question to ask the graph index."
    )
    return parser.parse_args()


def build_models() -> tuple[Ollama, HuggingFaceEmbedding]:
    llm = Ollama(model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))
    embed_model = HuggingFaceEmbedding(
        model_name=os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL),
        trust_remote_code=True,
        cache_folder=str(HF_CACHE_DIR),
    )
    return llm, embed_model


def load_documents() -> list:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    loader = SimpleDirectoryReader(str(DATA_DIR))
    documents = loader.load_data()
    if not documents:
        raise ValueError(f"No documents found in {DATA_DIR}")

    return documents


def build_index(
    documents: list, llm: Ollama, embed_model: HuggingFaceEmbedding
) -> PropertyGraphIndex:
    graph_store = Neo4jPropertyGraphStore(
        username=os.getenv("NEO4J_USERNAME", DEFAULT_NEO4J_USERNAME),
        password=os.getenv("NEO4J_PASSWORD", DEFAULT_NEO4J_PASSWORD),
        url=os.getenv("NEO4J_URL", DEFAULT_NEO4J_URL),
    )

    data_extractor = SimpleLLMPathExtractor(llm=llm)

    Settings.llm = llm
    Settings.embed_model = embed_model

    return PropertyGraphIndex.from_documents(
        documents,
        embed_model=embed_model,
        kg_extractors=[data_extractor],
        property_graph_store=graph_store,
        show_progress=True,
    )


def build_retriever(
    index: PropertyGraphIndex, llm: Ollama, embed_model: HuggingFaceEmbedding
):
    synonym_retriever = LLMSynonymRetriever(
        index.property_graph_store,
        llm=llm,
        include_text=False,
    )
    vector_retriever = VectorContextRetriever(
        index.property_graph_store,
        embed_model=embed_model,
        include_text=False,
    )

    return index.as_retriever(sub_retrievers=[synonym_retriever, vector_retriever])


def print_response(response) -> None:
    print("=" * 60)
    print(response.response)
    print("=" * 60)

    if not response.source_nodes:
        print("No source nodes returned.")
        print("=" * 60)
        return

    print(f"Source nodes: {len(response.source_nodes)}")
    print("=" * 60)
    for index, source_node in enumerate(response.source_nodes[:3], start=1):
        print(f"[{index}] {source_node.node.text}")
        print("-" * 60)


def main() -> None:
    args = parse_args()
    llm, embed_model = build_models()
    documents = load_documents()
    index = build_index(documents, llm, embed_model)
    retriever = build_retriever(index, llm, embed_model)

    # Warm up the retriever so both retrieval paths are exercised.
    retriever.retrieve("What is YC")

    query_engine = index.as_query_engine(include_text=True)
    response = query_engine.query(args.query)
    print_response(response)


if __name__ == "__main__":
    main()
