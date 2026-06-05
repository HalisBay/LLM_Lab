import torch


def search(model, processor, query, index, images, device, k=5):
    inputs = processor(text=[query], return_tensors="pt").to(device)

    with torch.no_grad():
        q = model.get_text_features(**inputs)

    # If model returned a ModelOutput, extract tensor
    if hasattr(q, "pooler_output"):
        q = q.pooler_output
    elif hasattr(q, "last_hidden_state") and q.last_hidden_state is not None:
        q = q.last_hidden_state.mean(dim=1)

    q = q / q.norm(dim=-1, keepdim=True)
    q = q.cpu().numpy().astype("float32")

    scores, ids = index.search(q, k)

    # return list of (index, score) pairs so callers can access images or metadata
    return [(int(ids[0][j]), float(scores[0][j])) for j in range(len(ids[0]))]
