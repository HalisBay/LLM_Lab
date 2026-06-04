import time

from qdrant_client import models


class Retriever:

    def __init__(
        self,
        vector_db,
        embeddata,
    ):
        self.vector_db = vector_db
        self.embeddata = embeddata

    def search(self, query):

        query_embedding = self.embeddata.embed_model.get_query_embedding(query)

        start_time = time.time()

        result = self.vector_db.client.query_points(
            collection_name=self.vector_db.collection_name,
            query=query_embedding,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=2.0,
                )
            ),
            timeout=1000,
        )

        elapsed_time = time.time() - start_time

        print(f"Execution time for the search: " f"{elapsed_time:.4f} seconds")

        return result
