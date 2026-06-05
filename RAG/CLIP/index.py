import faiss
import numpy as np
import torch


def build_index(model, processor, images, device):
    embeddings = []

    for img in images:
        inputs = processor(images=[img], return_tensors="pt").to(device)

        with torch.no_grad():
            emb = model.get_image_features(**inputs)

        # If model returned a ModelOutput, extract tensor
        if hasattr(emb, "pooler_output"):
            emb = emb.pooler_output
        elif hasattr(emb, "last_hidden_state") and emb.last_hidden_state is not None:
            emb = emb.last_hidden_state.mean(dim=1)

        emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu().numpy())

    embeddings = np.vstack(embeddings).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings
