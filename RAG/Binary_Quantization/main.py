import pickle

from embedder import EmbedData
from vectordb import QdrantVDB_BQ, QdrantVDB
from retriever import Retriever
from rag import RAG

embeddata = EmbedData(batch_size=32)

with open(
    "embeddings_and_contexts.pkl",
    "rb",
) as f:

    embeddings, contexts = pickle.load(f)

embeddata.embeddings = embeddings
embeddata.contexts = contexts

database = QdrantVDB("squad_collection")

database.define_client()
database.create_collection()
database.ingest_data(embeddata)

retriever = Retriever(
    database,
    embeddata,
)

rag = RAG(retriever)

query = (
    "The premium and VIP services in Airports "
    "are reserved for which type of passengers?"
)

answer = rag.query(query)

print(answer)

# without bq Execution time for the search: 3.1614 seconds

# with BQ Execution time for the search: 0.0244 seconds
