from rag_engine import get_query_engine

query_engine = get_query_engine()

response = query_engine.query("""
    How did the structure of funding startups
    in batches contribute to the success and
    growth of the Y Combinator program?
    """)
print(response)
