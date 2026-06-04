from llama_index.llms.ollama import Ollama


class RAG:

    def __init__(
        self,
        retriever,
        llm_name="qwen2.5:0.5b",
    ):

        self.retriever = retriever
        self.llm = Ollama(model=llm_name)

        self.qa_prompt_tmpl_str = """
Context information is below.

---------------------
{context}
---------------------

Given the context information above I want you
to think step by step to answer the query in a
crisp manner, incase case you don't know the
answer say 'I don't know!'

---------------------
Query: {query}
---------------------

Answer:
"""

    def generate_context(self, query):

        result = self.retriever.search(query)

        contexts = [point.payload["context"] for point in result.points]

        return "\n\n---\n\n".join(contexts)

    def query(self, query):

        context = self.generate_context(query)

        prompt = self.qa_prompt_tmpl_str.format(
            context=context,
            query=query,
        )

        response = self.llm.complete(prompt)

        return dict(response)["text"]
