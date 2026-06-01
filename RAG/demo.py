from pathlib import Path

import qdrant_client

try:
    # PyMuPDF (fitz) is used here because the default PDF reader was
    # pulling out broken / raw PDF structure instead of readable text.
    import fitz
except ImportError:
    fitz = None

from llama_index.core import Document

collection_name = "chat_with_docs_pymupdf"

BUILD_INDEX = False

# Local Qdrant server used as the vector database.
client = qdrant_client.QdrantClient(host="localhost", port=6333)

vector_store = None


doc_path = "./docs"


def load_docs():
    # Read every PDF page as a separate Document so retrieval can point to the
    # exact page that contains the answer, instead of treating the whole PDF as one blob.
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF is required for PDF extraction. Install pymupdf in the active environment."
        )

    docs = []
    pdf_paths = sorted(Path(doc_path).rglob("*.pdf"))

    for pdf_path in pdf_paths:
        pdf = fitz.open(pdf_path)
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue

            # Store file/page metadata so later we can inspect which PDF page
            # produced a given chunk.
            docs.append(
                Document(
                    text=text,
                    metadata={
                        "file_name": pdf_path.name,
                        "file_path": str(pdf_path),
                        "page_number": page_number,
                    },
                )
            )

    print(type(docs))
    print(len(docs))

    if docs:
        print("*" * 10, "--- SAMPLE DOC TEXT (first 500 chars) ---", "*" * 10)
        print(docs[0].text[:500].replace("\n", " "))
        print("*" * 50)

    return docs


from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext

# Qdrant is the persistent vector store. If the collection already exists,
# we can reuse it; otherwise we rebuild it from the PDFs.
vector_store = QdrantVectorStore(client=client, collection_name=collection_name)

collection_exists = client.collection_exists(collection_name=collection_name)


def create_index(docs):

    # This tells LlamaIndex where embeddings should be written.
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Convert raw documents into embedded chunks and push them into Qdrant.
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

    return index


from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# BAAI/bge-large-en-v1.5 is a strong general-purpose embedding model for
# semantic search and tends to work well for document retrieval.
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True
)

Settings.embed_model = embed_model

if BUILD_INDEX or not collection_exists:
    # BUILD_INDEX=True forces a fresh rebuild.
    # If the collection is missing, we also rebuild automatically so the script
    # does not crash on first run.
    if not BUILD_INDEX:
        print(f"Collection '{collection_name}' not found, building it from docs...")
    if BUILD_INDEX and collection_exists:
        # Clear the old vectors first so we do not mix stale and fresh content.
        client.delete_collection(collection_name=collection_name)
    docs = load_docs()
    index = create_index(docs)
else:
    # Reuse the existing collection without re-embedding every time.
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# debug: print collection info
try:
    if client.collection_exists(collection_name=collection_name):
        info = client.get_collection(collection_name)
        print(f"Collection points_count: {info.points_count}")
except Exception:
    pass


from llama_index.llms.ollama import Ollama

# Ollama runs the local LLM that turns retrieved context into the final answer.
llm = Ollama(model="llama3.2:1b", request_timeout=120.0)

Settings.llm = llm

from llama_index.core import PromptTemplate

# Keep the answer grounded in retrieved context. If the answer is not in the
# retrieved text, the model should say "I don't know!" instead of guessing.
template = """Context information is below:
              ---------------------
              {context_str}
              ---------------------
              Given the context information above I want you to think
              step by step to answer the query in a crisp manner,
              incase you don't know the answer say 'I don't know!'
            
              Query: {query_str}
        
              Answer:"""

qa_prompt_tmpl = PromptTemplate(template)

from llama_index.core.postprocessor import SentenceTransformerRerank

# Reranking is a second pass that reorders the top candidates from Qdrant and
# keeps the most relevant chunks near the top.
rerank = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3
)


# First retrieve a broader set of candidates, then let the reranker trim them
# down to the best few chunks.
query_engine = index.as_query_engine(similarity_top_k=10, node_postprocessors=[rerank])

# Swap in our stricter QA template so the response stays tied to the source text.
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

# Example question used as a quick smoke test for the full RAG pipeline.
response = query_engine.query("What is DSPy?")


print("*" * 25, "RESPONSE", "*" * 25)
print(str(response))
print("*" * 50)
print("\n" + "*" * 25, "SOURCE NODES", "*" * 25)
for i, node in enumerate(getattr(response, "source_nodes", [])[:5], 1):
    # Print the top retrieved chunks so we can inspect what the model actually saw.
    score = getattr(node, "score", None)
    text = getattr(getattr(node, "node", None), "text", "")
    print(f"{i}. score={score}")
    print(text[:400].replace("\n", " "))
    print("---")
print("*" * 50)
