from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

from datasets import Dataset
from tqdm.auto import tqdm
import pandas as pd

from rag_engine import query_engine

# ==========================================================
# Load + Split Docs
# ==========================================================

loader = DirectoryLoader("./docs/")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20)

documents = splitter.split_documents(documents)

print("chunks:", len(documents))

# ==========================================================
# Models
# ==========================================================

generator_llm = Ollama(model="qwen2.5:0.5b")
critic_llm = Ollama(model="qwen2.5:0.5b")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ==========================================================
# Testset Generator
# ==========================================================

generator = TestsetGenerator.from_langchain(
    generator_llm=generator_llm, critic_llm=critic_llm, embeddings=embeddings
)

distribution = {
    simple: 0.5,
    reasoning: 0.25,
    multi_context: 0.25,
}

testset = generator.generate_with_langchain_docs(
    documents, test_size=10, distributions=distribution, raise_exceptions=False
)

test_df = testset.to_pandas().dropna()

test_df.to_csv("test_data.csv", index=False)

# ==========================================================
# Build RAG evaluation dataset
# ==========================================================


def run_rag(q):
    res = query_engine.query(q)
    return {
        "answer": res.response,
        "contexts": [c.node.get_content() for c in res.source_nodes],
    }


questions = test_df["question"].values

responses = [run_rag(q) for q in tqdm(questions)]

dataset = Dataset.from_dict(
    {
        "question": questions,
        "answer": [r["answer"] for r in responses],
        "contexts": [r["contexts"] for r in responses],
        "ground_truth": test_df["ground_truth"].tolist(),
    }
)

# ==========================================================
# RAGAS Evaluation
# ==========================================================

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_recall,
    context_precision,
)

critic_llm = Ollama(model="qwen2.5:0.5b")

result = evaluate(
    llm=critic_llm,
    embeddings=embeddings,
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_correctness,
        context_recall,
        context_precision,
    ],
)

df = pd.DataFrame(result.scores)

print(df)
print("\nAVG:\n", df.mean())
