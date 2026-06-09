import torch
import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
from datasets import load_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device.type == "cuda" else torch.float32
model_device_map = "auto" if device.type == "cuda" else "cpu"


dataset = load_dataset("Pran10/ColpaliTest",
                       split="train",
                       cache_dir="./hf_cache")



print("Dataset image [3] loaded")
dataset[3]["image"].show()


import qdrant_client

client = qdrant_client.QdrantClient(
    host="localhost",
    port=6333
)

from colpali_engine.models import ColIdefics3, ColIdefics3Processor

model_name = "vidore/colSmol-256M"

colpali_model = ColIdefics3.from_pretrained(
    model_name,
    torch_dtype=model_dtype,
    device_map=model_device_map,
    trust_remote_code=True, 
    cache_dir="./hf_cache"
)

colpali_processor = ColIdefics3Processor.from_pretrained(model_name)


collection_name = "colpaili_demo"

if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    on_disk_payload=True,
    vectors_config=models.VectorParams(
        size=128,
        distance=models.Distance.COSINE,
        on_disk=True,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
    ),
)


batch_size = 1
all_embeddings = torch.zeros(len(dataset), 128)


total = len(dataset)
for i in tqdm(range(0, total, batch_size), desc="Indexing Progress", total=total//batch_size + 1):

    batch = dataset[i : i + batch_size]
    images = batch["image"]

    with torch.no_grad():
        batch_images = colpali_processor.process_images(images).to(device)
        image_embeddings = colpali_model(**batch_images)

    points = []
    for j, embedding in enumerate(image_embeddings):
            multivector = embedding.cpu().float().numpy().tolist()

            points.append(
                models.PointStruct(
                    id=i+j,  # we just use the index as the ID
                    vector=multivector,  # This is now a list of vectors
                )
            )

    try:
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=False,
        )
    except Exception as e:
        print(f"Error during upsert: {e}")
        continue
print("Indexing complete!")


query_text = "Which two players competed in the title fight?"

with torch.no_grad():
    query = colpali_processor.process_queries([query_text]).to(device)
    
    query_embedding = colpali_model(**query)

print("Query embedding shape:", query_embedding.shape)


token_query = query_embedding[0].cpu().float().numpy().tolist()

start_time = time.time()

query_result = client.query_points(collection_name=collection_name,
                                   query=token_query,
                                   limit=4,
                                   search_params=models.SearchParams(
                                   quantization=models.QuantizationSearchParams(
                                   ignore=True,
                                   rescore=True,
                                   oversampling=2.0
                                   )
                               )
                           )

print(f"Time taken = {(time.time()-start_time):.3f} s")

qa_prompt_tmpl_str = """The user has asked the following question:

                        ---------------------
                        
                        Query: {query}
                        
                        ---------------------

                        Some images are available to you
                        for this question. You have
                        to understand these images thoroughly and 
                        extract all relevant information that might 
                        help you answer the query better.
                                     
                        Given the context information above I want you
                        to think step by step to answer the query in a
                        crisp manner, in case you don't know the
                        answer say 'I don't know!'
                                     
                        ---------------------
                        Answer: """

prompt = qa_prompt_tmpl_str.format(query=query_text)

from io import BytesIO

def pil_image_to_jpeg_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()

output_idx = int(query_result.points[0].id)

print("\n=== RETRIEVAL RESULTS ===")

for point in query_result.points:
    print(
        f"ID={point.id} | Score={point.score:.4f}"
    )

print(f"\nSelected image index: {output_idx}")

retrieved_image = dataset[output_idx]["image"]

retrieved_image.show()

messages = [{
    "role": "user",
    "content": prompt,
    "images": [pil_image_to_jpeg_bytes(retrieved_image)]
}]

import  ollama

response = ollama.chat(
    model="qwen2.5vl:3b",
    messages=messages
)

print("\n=== QWEN ANSWER ===\n")
print(response.message.content)

