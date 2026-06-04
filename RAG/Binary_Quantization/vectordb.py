from qdrant_client import QdrantClient
from qdrant_client import models
from tqdm import tqdm

from utils import batch_iterate


class QdrantVDB:

    def __init__(
        self,
        collection_name,
        vector_dim=768,
        batch_size=512,
    ):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.batch_size = batch_size

    def define_client(self):
        self.client = QdrantClient(
            url="http://localhost:6333",
            prefer_grpc=True,
        )

    def create_collection(self):

        if not self.client.collection_exists(collection_name=self.collection_name):

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=models.Distance.DOT,
                    on_disk=True,
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=5,
                    indexing_threshold=0,
                ),
            )

    def ingest_data(self, embeddata):

        for batch_context, batch_embeddings in tqdm(
            zip(
                batch_iterate(
                    embeddata.contexts,
                    self.batch_size,
                ),
                batch_iterate(
                    embeddata.embeddings,
                    self.batch_size,
                ),
            ),
            total=len(embeddata.contexts) // self.batch_size,
            desc="Ingesting in batches",
        ):

            self.client.upload_collection(
                collection_name=self.collection_name,
                vectors=batch_embeddings,
                payload=[{"context": context} for context in batch_context],
            )

        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        )


class QdrantVDB_BQ(QdrantVDB):

    def create_collection(self):

        if not self.client.collection_exists(collection_name=self.collection_name):

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=models.Distance.DOT,
                    on_disk=True,
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=5,
                    indexing_threshold=0,
                ),
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(always_ram=True)
                ),
            )
